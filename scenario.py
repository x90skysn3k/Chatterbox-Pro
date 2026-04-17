"""
Phishing / vishing scenario runner for authorized offsec engagements.

Takes a YAML (or JSON) scenario describing a voice-channel call flow, renders
every step's narration through the standard /tts pipeline, and returns a
bundle of WAVs + manifest.json. Every run is tagged with an engagement_id
and written to `logs/phish-audit-YYYY-MM-DD.jsonl` for compliance.

By design this module DOES NOT:
  * place live calls — that lives behind the SIP bridge (sip_bridge.py)
  * bypass Whisper validation or watermarking defaults
  * support any "undetectability" mode — generation path is identical to the
    normal TTS flow, so audio fingerprints match everything else the server
    emits. If an engagement needs audio that defeats deepfake detectors,
    that is out of scope here.

Authorization model: the scenario MUST include non-empty `engagement_id` and
`authorization` fields. The server does not verify the authorization is
valid — it records it verbatim in the audit log. This is a safety net, not a
gate. You are responsible for scope.

Schema:

    engagement_id: ACME-2026-Q2                 # required
    authorization: |                            # required
      Written MSA signed 2026-03-15.
      Scope: voice phishing targeting IT helpdesk personnel.
      Contact: csirt@acme.example
    voice: executed-edge.wav                    # optional, defaults to DEFAULT_VOICE
    params:                                     # optional, any TTSRequest field
      temperature: 0.75
      exaggeration: 0.65
      cfg_weight: 0.4
    steps:
      - id: greeting                            # required; unique per scenario
        text: "Hi, this is John from IT."       # required
        notes: "Opener"                         # optional free text
      - id: pretext
        text: "We're seeing unusual logins..."
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("extended-server")

AUDIT_DIR = "logs"
os.makedirs(AUDIT_DIR, exist_ok=True)

_STEP_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


class ScenarioError(ValueError):
    """Validation or runtime error raised before or during a scenario run."""


@dataclass
class ScenarioStep:
    id: str
    text: str
    notes: str = ""
    # Per-step overrides on top of scenario.params
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    engagement_id: str
    authorization: str
    voice: str | None
    params: dict[str, Any]
    steps: list[ScenarioStep]
    name: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Scenario":
        if not isinstance(raw, dict):
            raise ScenarioError("Scenario must be a mapping")

        engagement_id = (raw.get("engagement_id") or "").strip()
        authorization = (raw.get("authorization") or "").strip()
        if not engagement_id:
            raise ScenarioError("engagement_id is required (authorized engagement tag)")
        if len(authorization) < 20:
            raise ScenarioError(
                "authorization is required — paste the scope / written authorization "
                "statement from your engagement letter (min 20 chars)"
            )

        steps_raw = raw.get("steps") or []
        if not isinstance(steps_raw, list) or not steps_raw:
            raise ScenarioError("steps[] must be a non-empty list")

        seen_ids = set()
        steps: list[ScenarioStep] = []
        for i, s in enumerate(steps_raw):
            if not isinstance(s, dict):
                raise ScenarioError(f"step[{i}] must be a mapping")
            sid = (s.get("id") or "").strip()
            if not sid or not _STEP_ID_RE.match(sid):
                raise ScenarioError(
                    f"step[{i}].id must match ^[a-zA-Z0-9_-]{{1,64}}$, got {sid!r}"
                )
            if sid in seen_ids:
                raise ScenarioError(f"duplicate step id: {sid}")
            seen_ids.add(sid)
            text = (s.get("text") or "").strip()
            if not text:
                raise ScenarioError(f"step {sid}: text is empty")
            if len(text) > 5000:
                raise ScenarioError(f"step {sid}: text over 5000 chars")
            steps.append(ScenarioStep(
                id=sid,
                text=text,
                notes=(s.get("notes") or "").strip(),
                params=s.get("params") or {},
            ))

        return cls(
            engagement_id=engagement_id,
            authorization=authorization,
            voice=raw.get("voice"),
            params=raw.get("params") or {},
            steps=steps,
            name=(raw.get("name") or "").strip(),
        )


def parse_scenario(body: bytes | str, content_type: str = "") -> Scenario:
    """Parse a scenario from YAML or JSON bytes/string."""
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    ct = (content_type or "").lower()
    data: Any
    if "yaml" in ct or body.lstrip().startswith(("---", "engagement_id", "name:")):
        try:
            import yaml
        except ImportError as e:
            raise ScenarioError("PyYAML required to parse YAML scenarios") from e
        try:
            data = yaml.safe_load(body)
        except yaml.YAMLError as e:
            raise ScenarioError(f"invalid YAML: {e}") from e
    else:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise ScenarioError(f"invalid JSON: {e}") from e
    return Scenario.from_dict(data)


def audit_log(record: dict[str, Any]) -> str:
    """Append one JSONL record to today's phishing audit log. Returns the file path."""
    path = os.path.join(AUDIT_DIR, f"phish-audit-{time.strftime('%Y-%m-%d')}.jsonl")
    record = dict(record)
    record.setdefault("ts", time.time())
    record.setdefault("ts_iso", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error(f"[scenario] audit log write failed: {e}")
    return path


# ---------- Runtime ----------

_runs: dict[str, dict[str, Any]] = {}
_runs_lock = threading.Lock()


def _run_id() -> str:
    return "scn_" + uuid.uuid4().hex[:10]


def start_run(scenario: Scenario, *, tts_submit: Callable[[dict[str, Any]], str],
              tts_wait: Callable[[str], dict[str, Any]],
              tts_fetch: Callable[[str], bytes]) -> str:
    """
    Kick off a scenario run in a background thread.

    The three callables keep this module decoupled from server.py:
      tts_submit(request_dict) -> job_id   # synchronous POST /tts equivalent
      tts_wait(job_id) -> final_status     # blocks until job terminal
      tts_fetch(job_id) -> wav bytes       # ?keep=false equivalent (we pull once)
    """
    run_id = _run_id()
    with _runs_lock:
        _runs[run_id] = {
            "run_id": run_id,
            "engagement_id": scenario.engagement_id,
            "status": "running",
            "started": time.time(),
            "total_steps": len(scenario.steps),
            "completed_steps": 0,
            "current_step": None,
            "bundle_path": None,
            "manifest": None,
            "error": None,
        }

    audit_log({
        "event": "scenario_start",
        "run_id": run_id,
        "engagement_id": scenario.engagement_id,
        "authorization": scenario.authorization,
        "step_count": len(scenario.steps),
        "steps": [{"id": s.id, "text_length": len(s.text)} for s in scenario.steps],
    })

    t = threading.Thread(
        target=_execute_run,
        args=(run_id, scenario, tts_submit, tts_wait, tts_fetch),
        daemon=True,
    )
    t.start()
    return run_id


def _execute_run(run_id: str, scenario: Scenario,
                 tts_submit: Callable, tts_wait: Callable, tts_fetch: Callable):
    manifest = {
        "run_id": run_id,
        "engagement_id": scenario.engagement_id,
        "authorization": scenario.authorization,
        "name": scenario.name,
        "started_at": time.time(),
        "steps": [],
    }
    step_wavs: list[tuple[str, bytes]] = []

    try:
        for step in scenario.steps:
            with _runs_lock:
                _runs[run_id]["current_step"] = step.id

            # Merge scenario-level params with per-step overrides
            params = dict(scenario.params or {})
            params.update(step.params or {})

            tts_req = {
                "text": step.text,
                "predefined_voice_id": scenario.voice or params.get("predefined_voice_id")
                                       or os.environ.get("DEFAULT_VOICE", "default.wav"),
                "engagement_id": scenario.engagement_id,
                **{k: v for k, v in params.items()
                   if k not in ("predefined_voice_id",)},
            }

            job_id = tts_submit(tts_req)
            audit_log({
                "event": "scenario_step_started",
                "run_id": run_id,
                "step_id": step.id,
                "job_id": job_id,
                "engagement_id": scenario.engagement_id,
            })

            final = tts_wait(job_id)
            if final.get("status") != "done":
                raise ScenarioError(
                    f"step {step.id} job {job_id} ended as {final.get('status')}: "
                    f"{final.get('error', '—')}"
                )

            wav_bytes = tts_fetch(job_id)
            step_wavs.append((f"{step.id}.wav", wav_bytes))

            manifest["steps"].append({
                "id": step.id,
                "notes": step.notes,
                "text": step.text,
                "job_id": job_id,
                "audio_file": f"{step.id}.wav",
                "audio_bytes": len(wav_bytes),
                "elapsed_seconds": final.get("elapsed"),
            })

            with _runs_lock:
                _runs[run_id]["completed_steps"] += 1

            audit_log({
                "event": "scenario_step_done",
                "run_id": run_id,
                "step_id": step.id,
                "job_id": job_id,
                "engagement_id": scenario.engagement_id,
                "bytes": len(wav_bytes),
            })

        # Build ZIP bundle
        manifest["finished_at"] = time.time()
        manifest["duration_seconds"] = round(manifest["finished_at"] - manifest["started_at"], 2)
        manifest["status"] = "done"

        bundle_path = os.path.join("output", f"{run_id}.zip")
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            for name, content in step_wavs:
                zf.writestr(f"audio/{name}", content)

        with _runs_lock:
            _runs[run_id]["status"] = "done"
            _runs[run_id]["bundle_path"] = bundle_path
            _runs[run_id]["manifest"] = manifest

        audit_log({
            "event": "scenario_done",
            "run_id": run_id,
            "engagement_id": scenario.engagement_id,
            "bundle_path": bundle_path,
            "step_count": len(scenario.steps),
            "duration_seconds": manifest["duration_seconds"],
        })

    except Exception as e:
        logger.error(f"[scenario {run_id}] failed: {e}", exc_info=True)
        with _runs_lock:
            _runs[run_id]["status"] = "failed"
            _runs[run_id]["error"] = str(e)
        audit_log({
            "event": "scenario_failed",
            "run_id": run_id,
            "engagement_id": scenario.engagement_id,
            "error": str(e),
        })


def get_run(run_id: str) -> dict[str, Any] | None:
    with _runs_lock:
        return dict(_runs.get(run_id, {})) or None


def list_runs(limit: int = 50, engagement_id: str | None = None) -> list[dict[str, Any]]:
    with _runs_lock:
        rows = list(_runs.values())
    if engagement_id:
        rows = [r for r in rows if r.get("engagement_id") == engagement_id]
    rows.sort(key=lambda r: r.get("started", 0), reverse=True)
    return [dict(r) for r in rows[:limit]]


def read_bundle(run_id: str) -> bytes | None:
    run = get_run(run_id)
    if not run or run.get("status") != "done":
        return None
    path = run.get("bundle_path")
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()
