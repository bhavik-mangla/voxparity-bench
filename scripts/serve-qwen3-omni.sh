#!/usr/bin/env bash
# Serve Qwen3-Omni-30B-A3B (Q4) locally with audio input on :8801.
# MoE with 3.3B active params - fast decode on Apple Silicon; ~20GB RAM resident.
# On 24GB Macs first raise the Metal wired limit (does not persist across reboots):
#   sudo sysctl iogpu.wired_limit_mb=22000
set -euo pipefail
cd "$(dirname "$0")/.."
exec llama-server \
  -m models/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf \
  --port "${VOXPARITY_LLAMACPP_PORT:-8801}" \
  -c 4096 --parallel 1 -b 512 -ub 512 --no-warmup
