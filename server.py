"""
Chatterbox Pro TTS Server
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
import gc
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import Response, HTMLResponse, JSONResponse, FileResponse
import json as _json
from pydantic import BaseModel

from logging.handlers import RotatingFileHandler

# Local modules (keep the long-generation hot path in server.py, side concerns here)
import auth
import jobs_db
import scenario as scenario_mod

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("logs/server.log", maxBytes=10*1024*1024, backupCount=3),
    ],
)
logger = logging.getLogger("extended-server")
logger.info("Logging to logs/server.log")

# Read version
_VERSION = "unknown"
_version_path = os.path.join(EXTENDED_DIR, "VERSION")
if os.path.exists(_version_path):
    with open(_version_path) as f:
        _VERSION = f.read().strip()
logger.info(f"Chatterbox Pro TTS Server v{_VERSION}")

app = FastAPI(title="Chatterbox Pro TTS Server")
app.middleware("http")(auth.api_key_middleware)

VOICES_DIR = os.path.join(EXTENDED_DIR, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Initialize durable job store. Marks any "processing" rows from a prior run
# as "interrupted" so the dashboard is honest about them.
_interrupted = jobs_db.init_db()

# Lock to prevent concurrent process_text_for_tts calls — do not remove.
# Concurrent jobs caused audio crackling on the P40 (commit 31ac1f9).
_generation_lock = threading.Lock()


def _force_vram_cleanup():
    """Aggressive VRAM cleanup between jobs to combat memory leaks."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"[VRAM] Post-cleanup: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
    except ImportError:
        pass

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
        if not any(marker in line for marker in ("[DET]", "[PROGRESS]", "[DENOISE]", "[VAD]", "[TRIM]", "[CACHE]", "auto-editor", "ffmpeg normalization", "ALL GENERATIONS", "composite_score", "Selected", "WARNING")):
            return
        # Forward quality pipeline output to log file for debugging
        # Strip ANSI color codes for clean log output
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
        logger.info(f"[{self.job_id}] {clean}")
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

            if "[VAD]" in line:
                job["stage"] = "vad-trimming"
            elif _DENOISE_RE.search(line):
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
        logger.info(f"Model loaded! (server v{_VERSION})")


_turbo_model = None
_turbo_lock = threading.Lock()

def _get_turbo_model():
    """Load Chatterbox Turbo model (350M, 1-step decoder) for streaming."""
    global _turbo_model
    if _turbo_model is None:
        with _turbo_lock:
            if _turbo_model is None:
                logger.info("Loading Chatterbox Turbo model...")
                try:
                    from chatterbox.tts import ChatterboxTTS
                    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
                    _turbo_model = ChatterboxTTS.from_pretrained_turbo(device=device)
                    if hasattr(_turbo_model, "eval"):
                        _turbo_model.eval()
                    logger.info("Turbo model loaded!")
                except Exception as e:
                    logger.error(f"Failed to load Turbo model: {e}")
                    logger.info("Falling back to standard model for streaming")
                    from Chatter import get_or_load_model
                    _turbo_model = get_or_load_model()
    return _turbo_model


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
    top_p: float = 0.8
    repetition_penalty: float = 2.0
    skip_normalization: bool = False
    use_silero_vad: bool = True
    num_candidates: int = 2
    max_attempts: int = 3
    whisper_threshold: float = 0.85
    # Optional tag for authorized-offsec engagements — recorded in jobs.db + audit log
    engagement_id: str | None = None


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


def _is_cancelled(job_id: str) -> bool:
    """Lock-free cancel check — called from the generation thread between chunks."""
    job = _jobs.get(job_id)
    return bool(job) and job.get("cancel_requested", False)


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

        # Quick pre-lock cancel check — if the user cancelled while we were queued,
        # don't even grab the generation lock.
        if _is_cancelled(job_id):
            with _jobs_lock:
                _jobs[job_id]["status"] = "cancelled"
            jobs_db.record_job_cancelled(job_id)
            logger.info(f"[{job_id}] Cancelled before start")
            return

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
                use_auto_editor=not request.use_silero_vad,
                ae_threshold=0.04,
                ae_margin=0.4,
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
                normalize_audio=not request.skip_normalization,
                normalize_method="ebu",
                normalize_level=-16,
                normalize_tp=-1.5,
                normalize_lra=11,
                num_candidates_per_chunk=request.num_candidates,
                max_attempts_per_candidate=request.max_attempts,
                bypass_whisper_checking=False,
                whisper_model_name="medium",
                enable_parallel=True,
                num_parallel_workers=2,
                use_longest_transcript_on_fail=True,
                sound_words_field="",
                use_faster_whisper=True,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                use_silero_vad=request.use_silero_vad,
                whisper_threshold=request.whisper_threshold,
                cancel_check=lambda jid=job_id: _is_cancelled(jid),
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
        with _jobs_lock:
            started_time = _jobs[job_id]["started"]
        elapsed = round(time.time() - started_time)
        logger.info(f"[{job_id}] Done: {job_output} ({file_size} bytes, {elapsed}s)")

        _generation_count += 1

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["output_path"] = job_output
            started_time = _jobs[job_id]["started"]
        jobs_db.record_job_done(job_id, job_output, round(time.time() - started_time))

    except Exception as e:
        msg = str(e)
        # Chatter.py raises RuntimeError("GENERATION_CANCELLED:...") on cancel
        if msg.startswith("GENERATION_CANCELLED"):
            logger.info(f"[{job_id}] Cancelled: {msg}")
            with _jobs_lock:
                _jobs[job_id]["status"] = "cancelled"
                _jobs[job_id]["error"] = msg
            jobs_db.record_job_cancelled(job_id)
        else:
            logger.error(f"[{job_id}] Generation failed: {e}", exc_info=True)
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = msg
                started_time = _jobs[job_id].get("started", time.time())
            jobs_db.record_job_failed(job_id, msg, round(time.time() - started_time))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        _force_vram_cleanup()


def _cleanup_old_jobs():
    """Remove jobs older than 2 hours and cleanup VRAM if idle."""
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
        active = sum(1 for j in _jobs.values() if j["status"] == "processing")
    if active == 0 and expired:
        _force_vram_cleanup()

    # Cleanup temp files older than 24 hours
    temp_cutoff = time.time() - 86400
    try:
        for f in os.listdir("temp"):
            fpath = os.path.join("temp", f)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < temp_cutoff:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
async def index():
    _cleanup_old_jobs()
    with _jobs_lock:
        active = sum(1 for j in _jobs.values() if j["status"] == "processing")
        done = sum(1 for j in _jobs.values() if j["status"] == "done")
    return (
        "<html><body>"
        f"<h1>Chatterbox Pro TTS Server v{_VERSION}</h1>"
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
        if vram["used_pct"] > 90:
            vram["warning"] = "VRAM usage above 90% — consider restarting"
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

    # Disk usage
    disk = {}
    try:
        temp_size = sum(os.path.getsize(os.path.join("temp", f)) for f in os.listdir("temp") if os.path.isfile(os.path.join("temp", f)))
        output_size = sum(os.path.getsize(os.path.join("output", f)) for f in os.listdir("output") if os.path.isfile(os.path.join("output", f)))
        disk = {"temp_mb": round(temp_size / 1024**2, 1), "output_mb": round(output_size / 1024**2, 1)}
    except Exception:
        pass

    return JSONResponse({
        "status": "healthy",
        "version": _VERSION,
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
        "disk": disk,
    })


def _submit_tts_internal(request: "TTSRequest") -> str:
    """
    In-process equivalent of POST /tts. Returns the job_id after the
    background thread has been kicked off. Used by the /tts endpoint
    AND by the scenario runner so both paths go through the exact same
    pipeline (lock, VRAM cleanup, whisper validation, persistence).

    Raises FileNotFoundError if the voice is missing.
    """
    ensure_model()
    _cleanup_old_jobs()

    voice_path = os.path.join(VOICES_DIR, request.predefined_voice_id)
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice not found: {request.predefined_voice_id}")

    job_id = str(uuid.uuid4())[:8]

    logger.info(f"[{job_id}] Queued: {len(request.text)} chars, model={request.model}, "
                f"exag={request.exaggeration}, cfg={request.cfg_weight}, temp={request.temperature}, "
                f"top_p={request.top_p}, rep_penalty={request.repetition_penalty}, "
                f"candidates={request.num_candidates}, attempts={request.max_attempts}, "
                f"vad={request.use_silero_vad}, skip_norm={request.skip_normalization}, "
                f"whisper_thresh={request.whisper_threshold}")

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "started": time.time(),
            "output_path": None,
            "error": None,
            "stage": "queued",
            "progress_chunk": 0,
            "progress_total": 0,
            "progress_pct": "0%",
            "current_chunk": 0,
            "current_text": "",
            "current_candidate": 0,
            "current_attempt": 0,
            "last_duration": 0,
            "skip_normalization": request.skip_normalization,
            "cancel_requested": False,
            "engagement_id": request.engagement_id,
            "text_preview": request.text[:120],
        }

    try:
        req_snapshot = {
            "text_length": len(request.text),
            "text_preview": request.text[:200],
            "voice": request.predefined_voice_id,
            "model": request.model,
            "temperature": request.temperature,
            "exaggeration": request.exaggeration,
            "cfg_weight": request.cfg_weight,
            "top_p": request.top_p,
            "repetition_penalty": request.repetition_penalty,
            "whisper_threshold": request.whisper_threshold,
            "num_candidates": request.num_candidates,
            "max_attempts": request.max_attempts,
        }
        jobs_db.record_job_created(
            job_id, _json.dumps(req_snapshot), kind="tts",
            engagement_id=request.engagement_id,
        )
    except Exception as e:
        logger.warning(f"[{job_id}] jobs_db create failed: {e}")

    thread = threading.Thread(
        target=_run_generation, args=(job_id, request, voice_path), daemon=True
    )
    thread.start()
    return job_id


def _wait_for_job(job_id: str, poll_interval: float = 1.0,
                  timeout: float = 3600) -> dict:
    """Block until a job reaches a terminal state. Used by scenario runner."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _jobs_lock:
            job = _jobs.get(job_id)
            status = job["status"] if job else None
            if job and status not in ("processing", "queued"):
                return {
                    "status": status,
                    "error": job.get("error"),
                    "elapsed": round(time.time() - job["started"], 1),
                    "output_path": job.get("output_path"),
                }
        # If the job fell out of _jobs (e.g. /result consumed it), check sqlite
        if not job:
            db_row = jobs_db.get_job(job_id)
            if db_row and db_row["status"] not in ("processing",):
                return {
                    "status": db_row["status"],
                    "error": db_row.get("error"),
                    "elapsed": db_row.get("elapsed_seconds"),
                    "output_path": db_row.get("output_path"),
                }
        time.sleep(poll_interval)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def _fetch_job_wav(job_id: str, *, consume: bool = True) -> bytes:
    """In-process /result: read WAV bytes. consume=True mirrors default delete behavior."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    output_path = (job or {}).get("output_path") if job else None
    if not output_path or not os.path.exists(output_path):
        db_row = jobs_db.get_job(job_id)
        output_path = (db_row or {}).get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise FileNotFoundError(f"no output WAV for job {job_id}")
    with open(output_path, "rb") as f:
        data = f.read()
    if consume:
        try:
            os.remove(output_path)
        except OSError:
            pass
        with _jobs_lock:
            _jobs.pop(job_id, None)
        jobs_db.record_job_deleted(job_id)
    return data


@app.post("/tts")
async def tts(request: TTSRequest):
    try:
        job_id = _submit_tts_internal(request)
    except FileNotFoundError as e:
        return Response(content=str(e), status_code=404)
    return JSONResponse({"job_id": job_id, "status": "processing"})


@app.post("/cancel/{job_id}")
async def cancel(job_id: str):
    """Request cancellation of a running job. Takes effect at next chunk
    boundary — won't interrupt mid-chunk generation. Returns current status."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        if job["status"] not in ("processing", "queued"):
            return JSONResponse(
                {"job_id": job_id, "status": job["status"], "note": "already terminal"}
            )
        job["cancel_requested"] = True
    logger.info(f"[{job_id}] Cancel requested")
    return JSONResponse({"job_id": job_id, "status": "cancel_requested"})


@app.get("/jobs")
async def list_jobs(limit: int = 50, engagement_id: str | None = None,
                    include_deleted: bool = False):
    """
    Dashboard feed. Merges the in-memory _jobs dict (has live progress) with
    the sqlite history (has finished/failed/cancelled/interrupted runs).
    In-memory entries take precedence when a job_id exists in both.
    """
    merged: dict[str, dict] = {}
    # Historical first
    for row in jobs_db.list_jobs(limit=limit, include_deleted=include_deleted,
                                 engagement_id=engagement_id):
        merged[row["job_id"]] = {
            "job_id": row["job_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "elapsed_seconds": row.get("elapsed_seconds"),
            "error": row.get("error"),
            "engagement_id": row.get("engagement_id"),
            "kind": row.get("kind", "tts"),
            "has_output": bool(row.get("output_path")) and os.path.exists(row.get("output_path") or ""),
            "source": "db",
        }
    # Live overrides
    with _jobs_lock:
        for jid, j in _jobs.items():
            if engagement_id and j.get("engagement_id") != engagement_id:
                continue
            merged[jid] = {
                "job_id": jid,
                "status": j["status"],
                "created_at": j["started"],
                "updated_at": time.time(),
                "elapsed_seconds": round(time.time() - j["started"], 1),
                "error": j.get("error"),
                "engagement_id": j.get("engagement_id"),
                "kind": "tts",
                "stage": j.get("stage"),
                "progress_pct": j.get("progress_pct"),
                "chunk": f"{j.get('progress_chunk', 0)}/{j.get('progress_total', 0)}"
                          if j.get("progress_total") else None,
                "text_preview": j.get("text_preview", ""),
                "has_output": bool(j.get("output_path")) and os.path.exists(j.get("output_path") or ""),
                "source": "live",
            }
    items = sorted(merged.values(), key=lambda r: r.get("created_at") or 0, reverse=True)
    return JSONResponse({"jobs": items[:limit], "count": len(items)})


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
async def result(job_id: str, keep: bool = False):
    """
    Download the WAV. Default behavior (keep=false) is unchanged from
    earlier versions: delete the file and job record on download. Pass
    ?keep=true to keep the file and job record — needed by the /studio
    dashboard so the same job can be re-downloaded / previewed.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        # Fallback: look in the durable store for a kept file from a prior run
        db_row = jobs_db.get_job(job_id)
        if db_row and db_row.get("output_path") and os.path.exists(db_row["output_path"]):
            with open(db_row["output_path"], "rb") as f:
                audio_bytes = f.read()
            return Response(content=audio_bytes, media_type="audio/wav")
        return Response(content="Job not found", status_code=404)
    if job["status"] != "done":
        return Response(content=f"Job not ready (status: {job['status']})", status_code=409)

    output_path = job["output_path"]
    if not output_path or not os.path.exists(output_path):
        return Response(content="Output file missing", status_code=500)

    with open(output_path, "rb") as f:
        audio_bytes = f.read()

    logger.info(f"[{job_id}] Served: {len(audio_bytes)} bytes (keep={keep})")

    if not keep:
        try:
            os.remove(output_path)
        except OSError:
            pass
        with _jobs_lock:
            del _jobs[job_id]
        jobs_db.record_job_deleted(job_id)

    headers = {}
    # Signal to client whether server-side normalization was applied
    if not job.get("skip_normalization", False):
        headers["X-Audio-Normalized"] = "ebu"
        headers["X-Audio-Loudnorm"] = "I=-16:TP=-1.5:LRA=11"
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


@app.websocket("/stream")
async def stream_tts(websocket: WebSocket):
    """WebSocket streaming TTS: sends audio chunks as they generate."""
    await websocket.accept()

    # Enforce API key for WebSocket — the HTTP middleware can't gate the upgrade.
    if not auth.check_ws_key(websocket):
        await websocket.send_json({"error": "unauthorized"})
        await websocket.close(code=4401)
        return

    try:
        # Receive request
        data = await websocket.receive_text()
        request = _json.loads(data)
        text = request.get("text", "")
        voice = request.get("voice", os.environ.get("DEFAULT_VOICE", "default.wav"))

        if not text.strip():
            await websocket.send_json({"error": "No text provided"})
            await websocket.close()
            return

        voice_path = os.path.join(VOICES_DIR, voice)
        if not os.path.exists(voice_path):
            await websocket.send_json({"error": f"Voice not found: {voice}"})
            await websocket.close()
            return

        logger.info(f"[STREAM] Starting: {len(text)} chars, voice={voice}")
        await websocket.send_json({"status": "generating", "text_length": len(text)})

        import torch
        import torchaudio
        import io
        from nltk.tokenize import sent_tokenize

        # Load model (Turbo preferred, falls back to standard)
        model = _get_turbo_model()

        # Split into small chunks for progressive streaming
        sentences = sent_tokenize(text)
        # Merge very short sentences
        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 < 200:
                current = (current + " " + s).strip() if current else s
            else:
                if current:
                    chunks.append(current)
                current = s
        if current:
            chunks.append(current)

        if not chunks:
            chunks = [text]

        await websocket.send_json({"status": "chunks", "count": len(chunks)})

        start_time = time.time()
        sample_rate = getattr(model, 'sr', 24000)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            chunk_start = time.time()
            try:
                # Generate with minimal params for speed
                # Build kwargs dynamically — Turbo model may not support all params
                gen_kwargs = {"audio_prompt_path": voice_path, "temperature": 0.8}
                import inspect
                sig = inspect.signature(model.generate)
                if "apply_watermark" in sig.parameters:
                    gen_kwargs["apply_watermark"] = False
                with torch.inference_mode():
                    wav = model.generate(chunk, **gen_kwargs)

                # Convert to PCM16 bytes
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)

                # Resample to 24kHz if needed
                if sample_rate != 24000:
                    wav = torchaudio.functional.resample(wav, sample_rate, 24000)

                # Convert to 16-bit PCM bytes
                pcm = (wav.squeeze().clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy().tobytes()

                chunk_time = round(time.time() - chunk_start, 2)

                # Send metadata then audio
                await websocket.send_json({
                    "status": "chunk",
                    "index": i,
                    "total": len(chunks),
                    "text": chunk[:80],
                    "duration_ms": len(pcm) // 2 * 1000 // 24000,
                    "gen_time": chunk_time,
                })
                await websocket.send_bytes(pcm)

                logger.info(f"[STREAM] Chunk {i+1}/{len(chunks)}: {chunk_time}s, {len(pcm)} bytes")

            except Exception as e:
                logger.error(f"[STREAM] Chunk {i} failed: {e}")
                await websocket.send_json({"error": f"Chunk {i} failed: {str(e)}"})

        elapsed = round(time.time() - start_time, 2)
        await websocket.send_json({"status": "done", "chunks": len(chunks), "elapsed": elapsed})
        logger.info(f"[STREAM] Done: {len(chunks)} chunks in {elapsed}s")

    except WebSocketDisconnect:
        logger.info("[STREAM] Client disconnected")
    except Exception as e:
        logger.error(f"[STREAM] Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.get("/stream-test")
async def stream_test():
    """Serve the streaming TTS test page."""
    html_path = os.path.join(EXTENDED_DIR, "stream.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<html><body><h1>stream.html not found</h1></body></html>")


@app.get("/studio")
async def studio():
    """Serve the full-control studio UI (parameter sliders, job dashboard, presets)."""
    html_path = os.path.join(EXTENDED_DIR, "studio.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<html><body><h1>studio.html not found</h1></body></html>")


@app.get("/voices")
async def list_voices():
    """List available voice files."""
    voices = [f for f in os.listdir(VOICES_DIR) if f.endswith(('.wav', '.mp3', '.flac'))]
    return JSONResponse({"voices": sorted(voices)})


@app.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...)):
    """Upload a voice reference WAV file."""
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        return Response(content="Only .wav, .mp3, .flac files allowed", status_code=400)
    # Sanitize filename
    import re as _re
    safe_name = _re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
    dest = os.path.join(VOICES_DIR, safe_name)
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    logger.info(f"Voice uploaded: {safe_name} ({len(content)} bytes)")
    return JSONResponse({"filename": safe_name, "size": len(content)})


# ============================================================================
# Phishing / vishing scenario runner (authorized offsec use)
# See scenario.py for schema + authorization model.
# ============================================================================

def _scenario_tts_submit(req_dict: dict) -> str:
    req = TTSRequest(**req_dict)
    return _submit_tts_internal(req)


def _scenario_tts_wait(job_id: str) -> dict:
    return _wait_for_job(job_id)


def _scenario_tts_fetch(job_id: str) -> bytes:
    # Bundles keep their own WAV copies in the ZIP — safe to consume here.
    return _fetch_job_wav(job_id, consume=True)


@app.post("/scenario/run")
async def scenario_run(request: Request):
    """
    Kick off a phishing scenario run. Body may be YAML or JSON; Content-Type
    is checked, with a fallback sniff on the body contents. Returns a run_id
    for polling.

    The scenario MUST include a non-empty engagement_id and a written
    authorization statement — see scenario.py for schema.
    """
    body = await request.body()
    ct = request.headers.get("content-type", "")
    try:
        scn = scenario_mod.parse_scenario(body, ct)
    except scenario_mod.ScenarioError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if scn.voice:
        voice_path = os.path.join(VOICES_DIR, scn.voice)
        if not os.path.exists(voice_path):
            return JSONResponse(
                {"error": f"voice not found: {scn.voice}"}, status_code=404
            )

    ensure_model()
    run_id = scenario_mod.start_run(
        scn,
        tts_submit=_scenario_tts_submit,
        tts_wait=_scenario_tts_wait,
        tts_fetch=_scenario_tts_fetch,
    )
    return JSONResponse({
        "run_id": run_id,
        "engagement_id": scn.engagement_id,
        "steps": len(scn.steps),
        "status": "running",
    })


@app.get("/scenario/{run_id}")
async def scenario_status(run_id: str):
    run = scenario_mod.get_run(run_id)
    if not run:
        return JSONResponse({"error": "run not found"}, status_code=404)
    return JSONResponse(run)


@app.get("/scenario/{run_id}/bundle")
async def scenario_bundle(run_id: str):
    """Download the ZIP bundle of WAVs + manifest for a completed run."""
    data = scenario_mod.read_bundle(run_id)
    if data is None:
        return Response(content="bundle not ready or run not found", status_code=404)
    return Response(
        content=data, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'},
    )


@app.get("/scenarios")
async def scenario_list(limit: int = 50, engagement_id: str | None = None):
    return JSONResponse({"runs": scenario_mod.list_runs(limit=limit, engagement_id=engagement_id)})


# ============================================================================
# On-the-fly short video generation
# Calls the parent repo (../../) Remotion pipeline via a CLI script.
# The heavy lifting (Pexels fetch, subtitle rendering) is handled there.
# ============================================================================

class ShortRenderRequest(TTSRequest):
    # Additional knobs specific to the short-video pipeline. Everything
    # inherited from TTSRequest is forwarded verbatim to /tts.
    background_query: str | None = None      # Pexels photo search query
    title_text: str | None = None            # 1-3 word hook for the title card


@app.post("/render-short")
async def render_short(request: ShortRenderRequest):
    """
    Submit-poll internally: generate narration, then shell out to the parent
    repo's Remotion pipeline to assemble the video. Returns the job_id of
    the underlying TTS run; the final MP4 path is in the response once done.

    This endpoint blocks until the MP4 is ready. For a fire-and-forget flow
    use POST /tts directly and run the renderer yourself.
    """
    import subprocess

    try:
        job_id = _submit_tts_internal(request)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    final = _wait_for_job(job_id, timeout=1800)
    if final["status"] != "done":
        return JSONResponse(
            {"job_id": job_id, "status": final["status"], "error": final.get("error")},
            status_code=500,
        )

    # Pull the WAV but keep the job around — the renderer needs the path
    wav_bytes = _fetch_job_wav(job_id, consume=False)
    wav_path = os.path.abspath(os.path.join("output", f"short_{job_id}.wav"))
    with open(wav_path, "wb") as f:
        f.write(wav_bytes)

    # Parent repo hook — optional. If the script is missing, we stop here
    # and just return the WAV path so the user can render manually.
    parent_cli = os.path.abspath(os.path.join(EXTENDED_DIR, "..", "..", "tools", "render-short-adhoc.sh"))
    if not os.path.exists(parent_cli):
        logger.info(f"[{job_id}] parent render script not found; returning WAV only")
        return JSONResponse({
            "job_id": job_id,
            "status": "audio_only",
            "wav_path": wav_path,
            "note": f"parent renderer not installed at {parent_cli}",
        })

    cmd = [
        "bash", parent_cli,
        "--audio", wav_path,
        "--text", request.text,
        "--title", request.title_text or (request.text.split(".")[0][:60]),
        "--bg-query", request.background_query or "dark cityscape",
        "--out", os.path.abspath(os.path.join("output", f"short_{job_id}.mp4")),
    ]
    logger.info(f"[{job_id}] render-short: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return JSONResponse({"job_id": job_id, "status": "render_timeout"}, status_code=504)

    if proc.returncode != 0:
        return JSONResponse({
            "job_id": job_id,
            "status": "render_failed",
            "wav_path": wav_path,
            "stderr": proc.stderr[-2000:],
        }, status_code=500)

    mp4_path = os.path.abspath(os.path.join("output", f"short_{job_id}.mp4"))
    return JSONResponse({
        "job_id": job_id,
        "status": "done",
        "wav_path": wav_path,
        "mp4_path": mp4_path if os.path.exists(mp4_path) else None,
    })


# ============================================================================
# Audit log tail (read-only, useful for dashboard)
# ============================================================================

@app.get("/audit/tail")
async def audit_tail(date: str | None = None, limit: int = 200):
    """Tail today's (or an ISO-date) phishing audit log. Read-only."""
    if date and not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        return JSONResponse({"error": "date must be YYYY-MM-DD"}, status_code=400)
    d = date or time.strftime("%Y-%m-%d")
    path = os.path.join("logs", f"phish-audit-{d}.jsonl")
    if not os.path.exists(path):
        return JSONResponse({"entries": [], "path": path, "note": "no log for this date"})
    entries = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(_json.loads(line))
        except _json.JSONDecodeError:
            continue
    return JSONResponse({"entries": entries, "path": path, "count": len(entries)})


if __name__ == "__main__":
    import uvicorn
    ensure_model()
    uvicorn.run(app, host="0.0.0.0", port=8004, timeout_keep_alive=300)
