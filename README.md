# Chatterbox-TTS-Extended

Production TTS pipeline based on [Chatterbox-TTS](https://github.com/resemble-ai/chatterbox) with async job queue, multi-candidate Whisper validation, neural denoising, and automated quality comparison.

## Quick Start (Docker)

```bash
# Clone and start
git clone https://github.com/x90skysn3k/Chatterbox-TTS-Extended.git
cd Chatterbox-TTS-Extended

# Add your voice reference file
cp /path/to/your-voice.wav voices/default.wav

# Start server (requires NVIDIA GPU + Docker with nvidia-container-toolkit)
docker compose up -d

# Generate speech
curl -X POST http://localhost:8004/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world.", "predefined_voice_id": "default.wav"}' \
  | jq .job_id
# Returns: {"job_id": "abc12345", "status": "processing"}

# Poll status
curl http://localhost:8004/status/abc12345

# Download WAV when done
curl http://localhost:8004/result/abc12345 --output speech.wav
```

## Quick Start (Manual)

```bash
pip install -r requirements.txt
./install-patches.sh   # Apply local fixes over pip package
python3 server.py      # Starts on port 8004
```

---

## Model Support Matrix

| Model | Params | Speed (P40) | CFG/Exagg | Notes |
|-------|--------|-------------|-----------|-------|
| **standard** (default) | 500M | ~10-15 min/scene | Yes | Best quality, full control |
| **turbo** | 350M | ~2-3 min/scene | No | 2x faster, paralinguistic tags |
| **multilingual** | 500M | ~10-15 min/scene | Yes | 23 languages |

Select model via API parameter: `"model": "standard"` (or `turbo`, `multilingual`).

## GPU Compatibility

| GPU | Compute | Dtype | VRAM | Notes |
|-----|---------|-------|------|-------|
| Tesla P40 | 6.1 | float32 | 24GB | Tested, production |
| RTX 2080 Ti | 7.5 | float16 | 11GB | Volta/Turing, half VRAM |
| RTX 3090 | 8.6 | bfloat16 | 24GB | Ampere, fast + efficient |
| A100 | 8.0 | bfloat16 | 40/80GB | Datacenter, best perf |
| RTX 4090 | 8.9 | bfloat16 | 24GB | Ada Lovelace, fastest |
| H100 | 9.0 | bfloat16 | 80GB | Hopper, top tier |

Dtype is auto-detected based on GPU compute capability. Override with env var:
```bash
CHATTERBOX_DTYPE=float32 docker compose up   # Force float32
CHATTERBOX_DTYPE=bfloat16 docker compose up  # Force bfloat16
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_VOICE` | `default.wav` | Voice reference file in `voices/` |
| `CHATTERBOX_DTYPE` | `auto` | Force dtype: `auto`, `float32`, `float16`, `bfloat16` |

---

## Server API

Async job queue architecture: submit, poll, download. No long-lived HTTP connections.

### Endpoints

#### `POST /tts` — Submit generation job

Returns `{ "job_id": "abc12345", "status": "processing" }` instantly.

```json
{
  "text": "Your narration text here.",
  "voice_mode": "predefined",
  "predefined_voice_id": "default.wav",
  "temperature": 0.75,
  "exaggeration": 0.65,
  "cfg_weight": 0.4,
  "speed_factor": 1.0,
  "split_text": true,
  "chunk_size": 250,
  "seed": 0,
  "model": "standard",
  "apply_watermark": false
}
```

#### `GET /status/{job_id}` — Poll progress

Returns real-time generation progress:

```json
{
  "job_id": "abc12345",
  "status": "processing",
  "elapsed": 45,
  "stage": "generation",
  "chunk": "2/5",
  "chunk_pct": "40%",
  "candidate": "cand 1 attempt 1",
  "text": "Most traders lose money..."
}
```

#### `GET /result/{job_id}` — Download WAV

Returns binary WAV audio. Deletes job after download.

#### `GET /health` — Server health check

```json
{
  "status": "healthy",
  "uptime": "2h 15m 30s",
  "uptime_seconds": 8130,
  "model": {
    "loaded": true,
    "type": "standard"
  },
  "vram": {
    "allocated_mb": 4521.3,
    "reserved_mb": 5120.0,
    "max_allocated_mb": 6200.1,
    "total_mb": 16384.0,
    "device": "Tesla P40",
    "used_pct": 27.6
  },
  "jobs": { "active": 1, "done": 5, "failed": 0 },
  "generation_count": 42
}
```

---

## Audio Pipeline

```
Text Input
  |
  v
Text Preprocessing (lowercase, spacing, dot-letters, sound words)
  |
  v
Sentence Tokenization (NLTK) + Smart Batching
  |
  v
Per-Chunk Generation (2 candidates, deterministic seeds, parallel workers)
  |
  v
Whisper Validation (faster-whisper medium, fuzzy match > 0.85)
  |  retry up to 3x per candidate if failed
  v
Best Candidate Selection (shortest passing, or longest transcript fallback)
  |
  v
Audio Concatenation
  |
  v
pyrnnoise Denoising (neural noise reduction, 48kHz)
  |
  v
Auto-Editor (silence trim, threshold=0.02, margin=0.4s)
  |
  v
FFmpeg Loudnorm (EBU R128: -16 LUFS, -1.5 TP, 11 LRA)
  |
  v
Output WAV (192kHz)
```

---

## QA Model Comparison

Compare TTS models and pipeline configurations with standardized test sentences.

### Quick start

```bash
# Compare current model (quick: 3 tests)
./run.sh tts-compare-quick

# Compare specific models
./run.sh tts-compare current,turbo,hf-800m

# Full comparison (all 7 tests)
npm run qa:tts-compare -- --models current,turbo --voice default.wav

# Use local M1 Max
npm run qa:tts-compare -- --models current --local --quick
```

### What it produces

```
tools/chatterbox-extended/qa-compare/
  2026-04-03T14-30-00/
    report.html          <-- Open this! Audio players + metrics side-by-side
    metrics/results.json
    audio/
      current/
        simple-narration.wav
        numbers-and-stats.wav
        ...
      turbo/
        ...
```

### Test corpus

7 standardized sentences in `test-corpus.json` covering:
- Simple narration (prosody baseline)
- Numbers and stats (pronunciation)
- Proper names (trader names like Livermore, Tudor Jones)
- Emotional range (somber tone, natural pauses)
- Technical terms (VIX, S&P, algo jargon)
- Short punchy sentences (rhythm, emphasis)
- Long complex sentences (breath pacing, dramatic build)

### Metrics computed

| Metric | Method | Pass threshold |
|--------|--------|----------------|
| Text accuracy | Whisper transcription + fuzzy match | > 0.85 |
| Duration | ffprobe | within 10% expected |
| Generation speed | wall clock | lower = better |
| Peak level | ffmpeg astats | -6 to -0.5 dBFS |
| File size | stat | informational |

---

## Deploy to P40 for Testing

Rapid iteration without git commits:

```bash
# Edit locally, deploy to P40, restart server
./tools/chatterbox-extended/deploy-p40.sh --restart

# Deploy + install new deps
./tools/chatterbox-extended/deploy-p40.sh --restart --deps

# Deploy + tail logs
./tools/chatterbox-extended/deploy-p40.sh --restart --logs

# Just sync files (no restart)
./tools/chatterbox-extended/deploy-p40.sh
```

The deploy script is gitignored. Workflow:
1. Edit files locally
2. `./deploy-p40.sh --restart`
3. `./run.sh tts-compare-quick`
4. Listen to `report.html`, iterate
5. When happy: `git add && git commit`

---

## Voice Settings (Production)

| Parameter | Value | Notes |
|-----------|-------|-------|
| temperature | 0.75 | Natural variance |
| exaggeration | 0.65 | Authoritative calm |
| cfg_weight | 0.4 | Balanced delivery |
| speed_factor | 1.0 | Must be 1.0 (other values cause double voice) |
| voice | default.wav | Custom reference |
| whisper_model | medium | Best accuracy/speed tradeoff |
| candidates | 2 | Per chunk |
| max_attempts | 3 | Retries per candidate |

---

## Infrastructure

### P40 Server (Primary)

```bash
# Server managed by systemd
ssh root@your-server "systemctl restart chatterbox-extended"
ssh root@your-server "journalctl -u chatterbox-extended -f"

# Check VRAM
ssh root@your-server "nvidia-smi"

# Health check
curl http://your-server:8004/health
```

### M1 Max (Fallback)

```bash
cd tools/chatterbox-extended
source venv/bin/activate
KMP_DUPLICATE_LIB_OK=TRUE python server.py
# Runs on port 8004 locally
```

Differences: `use_faster_whisper=False` (needs CUDA), `num_parallel_workers=1` (MPS limitation).

---

## Known Issues

### Memory leak on P40
VRAM usage grows over time. Restart after every 2-3 videos:
```bash
ssh root@your-server "systemctl restart chatterbox-extended"
```
The `/health` endpoint reports `vram.used_pct` — restart when approaching 80%.

### faster-whisper crashes
Occasionally silently crashes on certain audio. Falls back to longest-transcript selection. Switch to OpenAI Whisper if persistent: set `use_faster_whisper=False`.

---

## Dependencies

- Python 3.10.x
- FFmpeg on PATH
- CUDA 12.8 (P40) or MPS (M1 Max)
- PyTorch 2.7.0
- faster-whisper + openai-whisper
- pyrnnoise 0.3.8
- auto-editor 27.1.1

Full list: `requirements.txt`

---

## File Structure

```
server.py              FastAPI async job queue server
Chatter.py             Main TTS pipeline (batching, validation, post-processing)
test-corpus.json       QA comparison test sentences
deploy-p40.sh          P40 deployment script (gitignored)
voices/                Reference audio files
  default.wav    Production voice
chatterbox/src/        Core model implementations
  chatterbox/tts.py    ChatterboxTTS class (500M)
  chatterbox/vc.py     ChatterboxVC (voice conversion)
  models/t3/           T3 language model
  models/s3gen/        S3Gen vocoder
  models/voice_encoder/ Speaker embedding
  models/s3tokenizer/  Speech tokenizer
temp/                  Temporary candidate WAVs (gitignored)
output/                Final output files (gitignored)
qa-compare/            Comparison run results (gitignored)
```
