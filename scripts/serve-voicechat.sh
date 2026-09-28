#!/usr/bin/env bash
# Serve NVIDIA NemotronLabs VoiceChat 11B (OpenMDW-1.1; MLX 4-bit) on :8807 as a
# file-replay arm. Driver: `--driver llamacpp:nemotron-voicechat-11b-4bit`.
# mlx-vlm pinned in an isolated uv env. ~9 GB weights; no sysctl needed.
# VOXPARITY_VOICECHAT_TAIL_S sets the trailing silence appended to each clip
# (default 8 s; NVIDIA's offline recipe needs "sufficient trailing silence").
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-project --python 3.12 \
  --with 'mlx-vlm==0.7.3' --with scipy \
  python scripts/serve_voicechat.py --port "${VOXPARITY_VOICECHAT_PORT:-8807}" "$@"
