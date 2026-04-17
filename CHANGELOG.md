# Changelog

## [1.3.0] - 2026-04-17

### Added
- **Studio UI** (`studio.html` at `/studio`): full-control web interface with
  parameter sliders (temperature / exaggeration / cfg_weight / top_p /
  repetition_penalty / whisper_threshold), model switcher, voice dropdown
  with upload, live job dashboard polling `/jobs`, preset save/load via
  `localStorage`, engagement-id filter. Safe DOM construction — no
  `innerHTML` on untrusted strings.
- **Opt-in API key auth** (`auth.py`): `CHATTERBOX_API_KEY` env var gates
  every route except `/health`, `/`, `/stream-test`, `/studio`. WebSocket
  handshake gated via `check_ws_key`. Off by default — LAN behavior
  preserved.
- **SQLite job persistence** (`jobs_db.py`, `output/jobs.db`): mirrors
  lifecycle events (create, done, failed, cancelled, deleted). On startup
  marks any still-`processing` rows as `interrupted`. The in-memory
  `_jobs` dict remains the hot-path source of truth — sqlite never blocks
  generation.
- **POST `/cancel/{job_id}`**: flag-based cancel that takes effect at chunk
  boundaries via a new `cancel_check` callable plumbed through
  `process_text_for_tts`.
- **GET `/jobs`**: dashboard feed. Merges live `_jobs` (with progress)
  with sqlite history. Supports `engagement_id` filter and
  `include_deleted`.
- **`TTSRequest.whisper_threshold`** (default 0.85): replaces the two
  hardcoded `>= 0.85` checks in `Chatter.py` — initial and retry now use
  the same configurable value.
- **`TTSRequest.engagement_id`**: optional tag propagated to `_jobs`,
  `jobs.db`, and the audit log.
- **`/result/{job_id}?keep=true`**: retains file + job record so the
  dashboard can re-download. Default (auto-delete on download) unchanged.
- **Phishing / vishing scenario runner** (`scenario.py`): YAML- or
  JSON-defined call flows for authorized offsec engagements. Requires
  non-empty `engagement_id` and `authorization` (min 20 chars) — server
  records them in `logs/phish-audit-YYYY-MM-DD.jsonl` but does not
  verify. New endpoints: `POST /scenario/run`, `GET /scenario/{run_id}`,
  `GET /scenario/{run_id}/bundle` (ZIP with manifest + WAVs),
  `GET /scenarios`.
- **`GET /audit/tail`**: read-only tail of the phishing audit log
  (today or `?date=YYYY-MM-DD`).
- **SIP bridge scaffolding** (`sip_bridge.py`): abstract `AudioSink`
  with `FileSink` default and `SIPSink` stub. `SIPSink` requires
  `ENABLE_SIP_BRIDGE=1` and a pre-created named pipe — does not dial.
- **`POST /render-short`**: on-the-fly vertical video. Generates
  narration, shells `../render-short-adhoc.sh` for a 1080×1920 MP4.
  Graceful degradation to `status: "audio_only"` when the shell script
  is missing.
- **In-process TTS helpers**: `_submit_tts_internal`, `_wait_for_job`,
  `_fetch_job_wav` so scenario runner and `/render-short` share the
  exact same `_generation_lock` / pipeline as `POST /tts`.

### Changed
- **`uvicorn` → `uvicorn[standard]`** in `requirements.txt`. The
  `[standard]` extra pulls in `websockets`, which uvicorn needs to serve
  the existing `/stream` WebSocket. Without it, handshakes silently
  returned 404. Also added explicit `PyYAML` for the scenario parser.

### Invariants preserved
- `_generation_lock` still serializes — concurrent jobs caused audio
  crackling on the P40 (commit `31ac1f9`). Both new in-process
  submission paths go through it.
- `_jobs` dict remains the live-progress source of truth.
- `speed_factor` stays 1.0.
- `whisper_threshold` applies equally to initial and retry scoring.

## [Unreleased] - 2026-04-03

### Added
- **Async job queue server** (`server.py`): POST /tts returns job_id instantly, poll /status/{id}, download /result/{id}. Eliminates HTTP timeout issues during long generation runs.
- **`/health` endpoint**: Reports VRAM usage, GPU info (compute capability, dtype support), model state, job queue depth, uptime, generation count.
- **Model selection**: `model` parameter on `/tts` endpoint supports `standard` (500M), `turbo` (350M), `multilingual` (500M).
- **Watermark toggle**: `apply_watermark` parameter enables Perth neural watermarking (imperceptible, survives compression).
- **Pause tag support**: `[pause:Xs]` syntax in text — parsed, stripped before TTS, silence spliced into audio at correct positions.
- **Conditional caching**: Voice embeddings cached to `.conds.pt` with MD5 hash invalidation. Saves ~2s per generation call.
- **GPU dtype auto-detection**: Automatically selects bfloat16 (Ampere+), float16 (Volta/Turing), or float32 (Pascal) based on GPU compute capability. Override via `CHATTERBOX_DTYPE` env var.
- **QA model comparison**: `test-models.py` generates standardized test audio across models, produces HTML report with audio players for side-by-side listening.
- **Test corpus**: 7 standardized test sentences (`test-corpus.json`) covering easy/medium/hard difficulty levels.
- **Docker support**: Dockerfile, docker-compose.yml, .dockerignore for one-command deployment on any CUDA GPU.
- **GitHub Actions CI/CD**: Docker build+push to GHCR, lint + unit tests on every push/PR.
- **Install-patches script**: `install-patches.sh` overlays local tts.py fixes onto pip-installed chatterbox package.
- **Server file logging**: Logs to `/tmp/extended-server.log` for systemd/tail monitoring.

### Fixed
- **Memory leaks**: Explicit VRAM cleanup (del + gc.collect + cuda.empty_cache) after chunk candidates, concatenation, and generation completion. VRAM stable at ~3099MB across multiple generations on P40.
- **Turbo model dtype**: Monkey-patch for float64→float32 mismatch in s3tokenizer mel spectrogram (upstream bug in librosa→torch dtype conversion).
- **Multilingual model**: Added required `language_id` parameter for ChatterboxMultilingualTTS.generate().
- **Import compatibility**: Fallback from `chatterbox.src.chatterbox.tts` to `chatterbox.tts` for pip-installed package compatibility.
- **Generator kwarg**: Conditional pass to T3.inference() via inspect, handles pip versions that don't accept `generator` parameter.
- **tf32 on Ampere+**: Enable tf32 matmul on GPUs that support it (compute ≥ 8.0), disable on older GPUs for precision.

### Changed
- **Requirements**: Unpinned torch/torchaudio versions for cross-platform compatibility. Added `chatterbox-tts>=0.1.7` as primary dependency.
- **VRAM logging**: `_free_vram()` now accepts a label and logs current VRAM allocation at key pipeline stages.
