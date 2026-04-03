#!/bin/bash
# Start Extended server locally on M1 Max — port 8006
# Same server.py app, different port. Does NOT modify server.py or Chatter.py.
# Applies runtime patches for pip chatterbox compatibility.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

source venv/bin/activate

echo "Starting local Extended server on port 8006 (M1 Max MPS)..."
python -c "
import sys
sys.path.insert(0, '.')

# Runtime patches for pip chatterbox 0.1.7 compatibility
# (Same patches as P40, applied without modifying files)

# Patch 1: Fix import path (chatterbox.src.chatterbox → chatterbox)
import importlib
import types

# Create the missing module path so 'from chatterbox.src.chatterbox.tts import ...' works
import chatterbox
if not hasattr(chatterbox, 'src'):
    src_mod = types.ModuleType('chatterbox.src')
    src_inner = types.ModuleType('chatterbox.src.chatterbox')
    src_mod.chatterbox = src_inner
    chatterbox.src = src_mod
    sys.modules['chatterbox.src'] = src_mod
    sys.modules['chatterbox.src.chatterbox'] = src_inner

    # Map the actual modules
    from chatterbox import tts
    src_inner.tts = tts
    sys.modules['chatterbox.src.chatterbox.tts'] = tts

    try:
        from chatterbox import vc
        src_inner.vc = vc
        sys.modules['chatterbox.src.chatterbox.vc'] = vc
    except ImportError:
        pass

# Patch 2: Remove apply_watermark from ChatterboxTTS.generate if not supported
from chatterbox.tts import ChatterboxTTS
import inspect
gen_sig = inspect.signature(ChatterboxTTS.generate)
if 'apply_watermark' not in gen_sig.parameters:
    # Monkey-patch Chatter.py's calls by making apply_watermark a no-op kwarg
    _original_generate = ChatterboxTTS.generate
    def _patched_generate(self, *args, **kwargs):
        kwargs.pop('apply_watermark', None)
        return _original_generate(self, *args, **kwargs)
    ChatterboxTTS.generate = _patched_generate
    print('[PATCH] Removed apply_watermark from generate()')

import server
import uvicorn

# Override: faster-whisper doesn't support MPS, use OpenAI Whisper on M1
# Monkey-patch the _run_generation to force use_faster_whisper=False
_orig_run = server._run_generation
def _patched_run(job_id, request, voice_path):
    # Temporarily patch process_text_for_tts to override faster_whisper
    from Chatter import process_text_for_tts as _orig_pttf
    import functools
    @functools.wraps(_orig_pttf)
    def _wrapped(*args, **kwargs):
        kwargs['use_faster_whisper'] = False
        kwargs['num_parallel_workers'] = 1  # MPS can't handle parallel
        # Force Whisper to CPU (MPS has sparse tensor error with medium model)
        import Chatter
        _orig_load_whisper = Chatter.load_whisper_backend
        def _cpu_whisper(model_name, use_faster_whisper, device):
            return _orig_load_whisper(model_name, False, 'cpu')
        Chatter.load_whisper_backend = _cpu_whisper
        return _orig_pttf(*args, **kwargs)
    import Chatter
    Chatter.process_text_for_tts = _wrapped
    return _orig_run(job_id, request, voice_path)
server._run_generation = _patched_run
print('[PATCH] Forced use_faster_whisper=False + num_parallel_workers=1 for MPS')

server.ensure_model()
uvicorn.run(server.app, host='0.0.0.0', port=8006, timeout_keep_alive=300)
"
