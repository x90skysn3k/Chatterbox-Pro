# Changelog

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
