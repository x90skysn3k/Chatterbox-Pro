#!/bin/bash
# Overlay local patches onto pip-installed chatterbox package.
# Our tts.py has fixes (conditional caching, memory cleanup, watermark,
# generator kwarg compat) that need to replace the pip version.
#
# Safe to run multiple times (idempotent).
# Called automatically by Dockerfile and deploy-p40.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find pip-installed chatterbox package
SITE_PACKAGES=$(python3 -c "import chatterbox; import os; print(os.path.dirname(chatterbox.__file__))" 2>/dev/null)

if [ -z "$SITE_PACKAGES" ] || [ ! -d "$SITE_PACKAGES" ]; then
    echo "[patches] ERROR: chatterbox package not found. Install with: pip install chatterbox-tts"
    exit 1
fi

echo "[patches] Overlaying local patches onto $SITE_PACKAGES"

# Patch tts.py (conditional caching, memory cleanup, watermark, generator compat)
if [ -f "$SCRIPT_DIR/chatterbox/src/chatterbox/tts.py" ]; then
    cp "$SCRIPT_DIR/chatterbox/src/chatterbox/tts.py" "$SITE_PACKAGES/tts.py"
    echo "[patches]   tts.py → applied (caching, memory, watermark, generator)"
fi

echo "[patches] Done. All patches applied."
