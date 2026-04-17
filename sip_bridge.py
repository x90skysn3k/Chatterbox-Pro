"""
SIP / softphone bridge — stubbed.

Placing a *live* phone call through this server requires an external SIP
agent (Baresip, Asterisk originate, Linphone CLI, etc.) because the FastAPI
process is not going to route RTP itself. What this module provides:

  1. An `AudioSink` interface that decouples "audio was produced" from "audio
     was delivered". Default sink (`FileSink`) writes WAVs to disk. The
     placeholder `SIPSink` shells out to `baresip` via a named pipe — only
     available when the operator explicitly enables it.

  2. A hard safety gate: SIPSink refuses to initialize unless
     `ENABLE_SIP_BRIDGE=1` is set in the environment. Shipping this repo
     with a working dialer-by-default would be reckless.

  3. An integration surface for scenario.py (and future call-flow code) so
     the scenario runner can optionally pipe step audio into a live call
     without reaching outside this module.

For actual red-team engagements:
  - Caller-ID spoofing is handled by the SIP trunk / carrier, NOT here.
  - Legal scope, engagement_id, and written authorization live in the
    scenario file (see scenario.py) — this module does not re-verify them.
  - Every live call should be echoed to the phishing audit log.

If your engagement uses a different softphone (Asterisk AMI, Twilio Voice,
etc.), subclass AudioSink and register it in `SINK_REGISTRY`. Keep the SIP
details out of scenario.py so the authorization model stays uniform.
"""
from __future__ import annotations

import logging
import os
import subprocess
from abc import ABC, abstractmethod

logger = logging.getLogger("extended-server")

SIP_ENABLED = os.environ.get("ENABLE_SIP_BRIDGE", "").strip() == "1"


class AudioSink(ABC):
    @abstractmethod
    def deliver(self, step_id: str, wav_bytes: bytes, meta: dict) -> dict:
        """Deliver rendered audio. Return a dict of delivery-side metadata."""


class FileSink(AudioSink):
    """Default: write each step's WAV to disk. Used by scenario runner."""

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

    def deliver(self, step_id: str, wav_bytes: bytes, meta: dict) -> dict:
        path = os.path.join(self.out_dir, f"{step_id}.wav")
        with open(path, "wb") as f:
            f.write(wav_bytes)
        return {"path": path, "bytes": len(wav_bytes)}


class SIPSink(AudioSink):
    """
    Plays audio into a live call via an external Baresip control socket.

    Protocol assumed:
      - Baresip is already running with a registered SIP account.
      - Operator has placed the call (e.g., `baresip -e "/dial sip:victim@..."`)
        before starting the scenario run. The sink does NOT dial for you —
        that's an intentional seam so dialing stays under manual control.
      - This sink writes the audio to a named pipe that Baresip's `ausrc pipe`
        module reads from (or equivalent).

    Misconfigure the pipe and `deliver()` raises — fail loud rather than
    silently dropping payloads during an engagement.
    """

    def __init__(self, pipe_path: str | None = None):
        if not SIP_ENABLED:
            raise RuntimeError(
                "SIP bridge is gated: set ENABLE_SIP_BRIDGE=1 in the environment "
                "to enable. Default is OFF."
            )
        self.pipe_path = pipe_path or os.environ.get(
            "SIP_AUDIO_PIPE", "/tmp/chatterbox-sip.pipe"
        )
        if not os.path.exists(self.pipe_path):
            raise RuntimeError(
                f"SIP audio pipe missing: {self.pipe_path}. Create it with "
                f"`mkfifo {self.pipe_path}` and point baresip at it, then retry."
            )

    def deliver(self, step_id: str, wav_bytes: bytes, meta: dict) -> dict:
        logger.info(f"[sip] delivering step {step_id} ({len(wav_bytes)} bytes)")
        # ffmpeg transcodes WAV → 16kHz s16le mono PCM → pipe.
        # Baresip's `ausrc pipe` expects raw PCM; transcoding here keeps the
        # scenario author oblivious to SIP codec details.
        proc = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "wav", "-i", "pipe:0",
                "-f", "s16le", "-ar", "16000", "-ac", "1", self.pipe_path,
            ],
            stdin=subprocess.PIPE,
        )
        try:
            proc.communicate(input=wav_bytes, timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg → SIP pipe failed, rc={proc.returncode}")
        return {"sink": "sip", "pipe": self.pipe_path, "bytes": len(wav_bytes)}


SINK_REGISTRY: dict[str, type[AudioSink]] = {
    "file": FileSink,
    "sip": SIPSink,
}


def make_sink(kind: str, **kwargs) -> AudioSink:
    cls = SINK_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"unknown sink: {kind}")
    return cls(**kwargs)
