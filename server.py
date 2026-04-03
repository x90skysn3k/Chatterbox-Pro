"""
Chatterbox Extended TTS Server
Async job queue: POST /tts returns job_id instantly, poll /status/{id}, download /result/{id}.
Eliminates HTTP timeout issues during long generation runs.
Progress streamed via /status — client sees chunk/candidate/whisper progress in real time.

Models supported:
  - standard (500M) — Original Chatterbox, best quality with CFG/exaggeration control
  - hf-800m (800M) — Largest model, potentially better prosody (experimental)
  - turbo (350M) — Fastest, supports paralinguistic tags, no CFG/exaggeration
  - multilingual (500M) — 23 languages, emotion control
"""
import os
import sys
import re
import glob
import logging
import threading
import time
import uuid
import shutil

EXTENDED_DIR = os.path.dirname(os.path.abspath(__file__))
if EXTENDED_DIR not in sys.path:
    sys.path.insert(0, EXTENDED_DIR)

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from fastapi import FastAPI
from fastapi.responses import Response, HTMLResponse, JSONResponse
from pydantic import BaseModel

LOG_FILE = "/tmp/extended-server.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("extended-server")
logger.info(f"Logging to {LOG_FILE}")

app = FastAPI(title="Chatterbox Extended TTS Server")

VOICES_DIR = os.path.join(EXTENDED_DIR, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Lock to prevent concurrent process_text_for_tts calls (it purges temp/ dir)
_generation_lock = threading.Lock()

# Job store
_jobs = {}
_jobs_lock = threading.Lock()

# Pre-load model on startup
_model_loaded = False
_current_model_type = "standard"  # Track which model variant is loaded
_server_start_time = time.time()
_generation_count = 0  # Track total generations for memory leak monitoring

# Per-job progress capture — parses Extended's stdout for chunk/candidate/whisper info
_PROGRESS_RE = re.compile(r'\[PROGRESS\].*?(\d+)/(\d+).*?(\d+%)')
_CHUNK_RE = re.compile(r'\[DET\] Processing group (\d+):.*?len=\d+:(.*)')
_CAND_RE = re.compile(r'\[DET\] Generating cand (\d+) attempt (\d+) for chunk (\d+)')
_SAVED_RE = re.compile(r'\[DET\] Saved cand (\d+), attempt (\d+), duration=([\d.]+)s')
_WHISPER_RE = re.compile(r'whisper.*?score.*?([\d.]+)', re.IGNORECASE)
_DENOISE_RE = re.compile(r'\[DENOISE\]')
_AUTOEDITOR_RE = re.compile(r'auto-editor')
_NORMALIZE_RE = re.compile(r'ffmpeg normalization')
_COMPLETE_RE = re.compile(r'ALL GENERATIONS COMPLETE')


class TeeWriter:
    """Captures writes to both the original stream and a per-job ring buffer."""
    def __init__(self, original, job_id):
        self.original = original
        self.job_id = job_id

    def write(self, text):
        self.original.write(text)
        if not text or not text.strip():
            return
        # Only parse lines with progress markers — skip tqdm spam to avoid lock contention
        line = text.strip()
        if not any(marker in line for marker in ("[DET]", "[PROGRESS]", "[DENOISE]", "auto-editor", "ffmpeg normalization", "ALL GENERATIONS")):
            return
        with _jobs_lock:
            job = _jobs.get(self.job_id)
            if not job:
                return

            # Extract progress info
            m = _PROGRESS_RE.search(line)
            if m:
                job["progress_chunk"] = int(m.group(1))
                job["progress_total"] = int(m.group(2))
                job["progress_pct"] = m.group(3)

            m = _CHUNK_RE.search(line)
            if m:
                job["current_chunk"] = int(m.group(1))
                job["current_text"] = m.group(2).strip()[:80]

            m = _CAND_RE.search(line)
            if m:
                job["current_candidate"] = int(m.group(1))
                job["current_attempt"] = int(m.group(2))

            m = _SAVED_RE.search(line)
            if m:
                job["last_duration"] = float(m.group(3))

            if _DENOISE_RE.search(line):
                job["stage"] = "denoising"
            elif _AUTOEDITOR_RE.search(line):
                job["stage"] = "auto-editor"
            elif _NORMALIZE_RE.search(line):
                job["stage"] = "normalizing"
            elif _COMPLETE_RE.search(line):
                job["stage"] = "complete"

    def flush(self):
        self.original.flush()

    def fileno(self):
        return self.original.fileno()

    def isatty(self):
        return False


def ensure_model():
    global _model_loaded
    if not _model_loaded:
        logger.info("Pre-loading Chatterbox model...")
        from Chatter import get_or_load_model
        get_or_load_model()
        _model_loaded = True
        logger.info("Model loaded!")


class TTSRequest(BaseModel):
    text: str
    voice_mode: str = "predefined"
    predefined_voice_id: str = os.environ.get("DEFAULT_VOICE", "default.wav")
    temperature: float = 0.75
    exaggeration: float = 0.65
    cfg_weight: float = 0.4
    speed_factor: float = 1.0
    split_text: bool = True
    chunk_size: int = 250
    seed: int = 0
    model: str = "standard"  # standard | turbo | multilingual | hf-800m
    apply_watermark: bool = False


def _find_output_wav(result):
    """Find WAV file from process_text_for_tts result."""
    output_path = None
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, str) and item.endswith(".wav") and os.path.exists(item):
                output_path = item
                break

    # Fallback: find most recent WAV in output/
    if not output_path:
        wav_files = sorted(glob.glob("output/api_output*.wav"), key=os.path.getmtime, reverse=True)
        if wav_files:
            output_path = wav_files[0]

    return output_path


def _run_generation(job_id, request, voice_path):
    """Background thread: runs process_text_for_tts and updates job status."""
    global _generation_count
    # Install TeeWriter to capture Extended's stdout/stderr progress
    old_stdout, old_stderr = sys.stdout, sys.stderr
    tee_out = TeeWriter(old_stdout, job_id)
    tee_err = TeeWriter(old_stderr, job_id)
    try:
        from Chatter import process_text_for_tts

        logger.info(f"[{job_id}] Starting generation: {len(request.text)} chars, model={request.model}")

        sys.stdout = tee_out
        sys.stderr = tee_err

        with _generation_lock:
            result = process_text_for_tts(
                text=request.text,
                input_basename=f"api_{job_id}",
                audio_prompt_path_input=voice_path,
                exaggeration_input=request.exaggeration,
                temperature_input=request.temperature,
                seed_num_input=request.seed,
                cfgw_input=request.cfg_weight,
                use_pyrnnoise=True,
                use_auto_editor=True,
                ae_threshold=0.02,  # Gentler than 0.06 — only cuts true silence, not quiet speech
                ae_margin=0.4,      # 0.4s margin (was 0.2) — protects word boundaries
                export_formats=["wav"],
                enable_batching=False,
                to_lowercase=False,
                normalize_spacing=True,
                fix_dot_letters=False,
                remove_reference_numbers=False,
                keep_original_wav=False,
                smart_batch_short_sentences=True,
                disable_watermark=not request.apply_watermark,
                num_generations=1,
                normalize_audio=True,
                normalize_method="ebu",
                normalize_level=-16,
                normalize_tp=-1.5,
                normalize_lra=11,
                num_candidates_per_chunk=2,
                max_attempts_per_candidate=3,
                bypass_whisper_checking=False,
                whisper_model_name="medium",
                enable_parallel=True,
                num_parallel_workers=2,
                use_longest_transcript_on_fail=True,
                sound_words_field="",
                use_faster_whisper=True,
            )

        output_path = _find_output_wav(result)
        if not output_path or not os.path.exists(output_path):
            logger.error(f"[{job_id}] No output WAV found. Result: {result}")
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = "No output WAV found"
            return

        # Move to job-specific path to avoid cleanup race
        job_output = f"output/job_{job_id}.wav"
        shutil.move(output_path, job_output)

        file_size = os.path.getsize(job_output)
        elapsed = round(time.time() - _jobs[job_id]["started"])
        logger.info(f"[{job_id}] Done: {job_output} ({file_size} bytes, {elapsed}s)")

        _generation_count += 1

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["output_path"] = job_output

    except Exception as e:
        logger.error(f"[{job_id}] Generation failed: {e}", exc_info=True)
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _cleanup_old_jobs():
    """Remove jobs older than 2 hours."""
    cutoff = time.time() - 7200
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items() if j["started"] < cutoff]
        for jid in expired:
            job = _jobs.pop(jid)
            if job.get("output_path") and os.path.exists(job["output_path"]):
                try:
                    os.remove(job["output_path"])
                except OSError:
                    pass
            logger.info(f"Cleaned up expired job {jid}")


@app.get("/", response_class=HTMLResponse)
async def index():
    _cleanup_old_jobs()
    with _jobs_lock:
        active = sum(1 for j in _jobs.values() if j["status"] == "processing")
        done = sum(1 for j in _jobs.values() if j["status"] == "done")
    return (
        "<html><body>"
        "<h1>Chatterbox Extended TTS Server (Async Job Queue)</h1>"
        f"<p>Active jobs: {active} | Completed: {done}</p>"
        "<p>POST /tts → returns job_id | GET /status/ID | GET /result/ID</p>"
        "</body></html>"
    )


@app.get("/health")
async def health():
    """Health check endpoint — reports VRAM, model state, job queue, uptime."""
    import torch

    uptime = round(time.time() - _server_start_time)
    uptime_str = f"{uptime // 3600}h {(uptime % 3600) // 60}m {uptime % 60}s"

    # VRAM + GPU stats (CUDA only)
    vram = {}
    gpu = {}
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        vram = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1),
            "device": torch.cuda.get_device_name(0),
        }
        vram["used_pct"] = round(vram["allocated_mb"] / vram["total_mb"] * 100, 1) if vram["total_mb"] > 0 else 0
        gpu = {
            "compute_capability": f"{cap[0]}.{cap[1]}",
            "supports_bf16": cap >= (8, 0),
            "supports_fp16": cap >= (7, 0),
            "supports_tf32": cap >= (8, 0),
        }

    # Job queue stats
    with _jobs_lock:
        active = sum(1 for j in _jobs.values() if j["status"] == "processing")
        done = sum(1 for j in _jobs.values() if j["status"] == "done")
        failed = sum(1 for j in _jobs.values() if j["status"] == "failed")

    return JSONResponse({
        "status": "healthy",
        "uptime": uptime_str,
        "uptime_seconds": uptime,
        "model": {
            "loaded": _model_loaded,
            "type": _current_model_type,
        },
        "vram": vram,
        "gpu": gpu,
        "jobs": {
            "active": active,
            "done": done,
            "failed": failed,
        },
        "generation_count": _generation_count,
    })


@app.post("/tts")
async def tts(request: TTSRequest):
    ensure_model()
    _cleanup_old_jobs()

    voice_path = os.path.join(VOICES_DIR, request.predefined_voice_id)
    if not os.path.exists(voice_path):
        return Response(content=f"Voice not found: {request.predefined_voice_id}", status_code=404)

    job_id = str(uuid.uuid4())[:8]

    logger.info(f"[{job_id}] Queued: {len(request.text)} chars, model={request.model}, "
                f"exag={request.exaggeration}, cfg={request.cfg_weight}, temp={request.temperature}")

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "started": time.time(),
            "output_path": None,
            "error": None,
            # Progress tracking (updated by TeeWriter parsing stdout)
            "stage": "queued",
            "progress_chunk": 0,
            "progress_total": 0,
            "progress_pct": "0%",
            "current_chunk": 0,
            "current_text": "",
            "current_candidate": 0,
            "current_attempt": 0,
            "last_duration": 0,
        }

    thread = threading.Thread(target=_run_generation, args=(job_id, request, voice_path), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id, "status": "processing"})


@app.get("/status/{job_id}")
async def status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    resp = {
        "job_id": job_id,
        "status": job["status"],
        "elapsed": round(time.time() - job["started"]),
    }
    if job["status"] == "done" and job.get("output_path") and os.path.exists(job["output_path"]):
        resp["file_size"] = os.path.getsize(job["output_path"])
    if job["status"] == "failed":
        resp["error"] = job.get("error", "Unknown error")

    # Progress details
    if job["status"] == "processing":
        resp["stage"] = job.get("stage", "queued")
        if job.get("progress_total"):
            resp["chunk"] = f"{job['progress_chunk']}/{job['progress_total']}"
            resp["chunk_pct"] = job.get("progress_pct", "0%")
        if job.get("current_text"):
            resp["text"] = job["current_text"]
        if job.get("current_candidate"):
            resp["candidate"] = f"cand {job['current_candidate']} attempt {job['current_attempt']}"
        if job.get("last_duration"):
            resp["last_chunk_dur"] = f"{job['last_duration']:.1f}s"

    return JSONResponse(resp)


@app.get("/result/{job_id}")
async def result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return Response(content="Job not found", status_code=404)
    if job["status"] != "done":
        return Response(content=f"Job not ready (status: {job['status']})", status_code=409)

    output_path = job["output_path"]
    if not output_path or not os.path.exists(output_path):
        return Response(content="Output file missing", status_code=500)

    with open(output_path, "rb") as f:
        audio_bytes = f.read()

    logger.info(f"[{job_id}] Served: {len(audio_bytes)} bytes")

    # Cleanup
    try:
        os.remove(output_path)
    except OSError:
        pass
    with _jobs_lock:
        del _jobs[job_id]

    return Response(content=audio_bytes, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    ensure_model()
    uvicorn.run(app, host="0.0.0.0", port=8004, timeout_keep_alive=300)
