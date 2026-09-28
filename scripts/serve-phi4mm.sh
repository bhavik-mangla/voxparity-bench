#!/usr/bin/env bash
# Serve Phi-4-multimodal-instruct (MIT; bf16 via MLX) for VoxParity on :8804.
# Driver: `--driver llamacpp:phi-4-multimodal-instruct` (tools in <|tool|> tokens).
# mlx-vlm is pinned in an isolated uv env (not a project dependency). ~12 GB
# resident in bf16; no iogpu.wired_limit_mb sysctl needed on 24 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-project --python 3.12 \
  --with 'mlx-vlm==0.7.3' --with scipy \
  python scripts/serve_phi4mm.py --port "${VOXPARITY_PHI4MM_PORT:-8804}" "$@"
