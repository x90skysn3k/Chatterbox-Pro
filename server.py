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

from fastapi import FastAPI
from fastapi.responses import Response, HTMLResponse, JSONResponse
from pydantic import BaseModel

from logging.handlers import RotatingFileHandler

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
logger.info(f"Chatterbox Extended TTS Server v{_VERSION}")

app = FastAPI(title="Chatterbox Extended TTS Server")

VOICES_DIR = os.path.join(EXTENDED_DIR, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Lock to prevent concurrent process_text_for_tts calls (it purges temp/ dir)
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

    except Exception as e:
        logger.error(f"[{job_id}] Generation failed: {e}", exc_info=True)
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
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
        f"<h1>Chatterbox Extended TTS Server v{_VERSION}</h1>"
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


@app.post("/tts")
async def tts(request: TTSRequest):
    ensure_model()
    _cleanup_old_jobs()

    voice_path = os.path.join(VOICES_DIR, request.predefined_voice_id)
    if not os.path.exists(voice_path):
        return Response(content=f"Voice not found: {request.predefined_voice_id}", status_code=404)

    job_id = str(uuid.uuid4())[:8]

    logger.info(f"[{job_id}] Queued: {len(request.text)} chars, model={request.model}, "
                f"exag={request.exaggeration}, cfg={request.cfg_weight}, temp={request.temperature}, "
                f"top_p={request.top_p}, rep_penalty={request.repetition_penalty}, "
                f"candidates={request.num_candidates}, attempts={request.max_attempts}, "
                f"vad={request.use_silero_vad}, skip_norm={request.skip_normalization}")

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
            "skip_normalization": request.skip_normalization,
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

    headers = {}
    # Signal to client whether server-side normalization was applied
    if not job.get("skip_normalization", False):
        headers["X-Audio-Normalized"] = "ebu"
        headers["X-Audio-Loudnorm"] = "I=-16:TP=-1.5:LRA=11"
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn
    ensure_model()
    uvicorn.run(app, host="0.0.0.0", port=8004, timeout_keep_alive=300)
