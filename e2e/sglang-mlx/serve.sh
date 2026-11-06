#!/usr/bin/env bash
set -euo pipefail

project=$(dirname "$(realpath "$0")")
export SGLANG_USE_MLX=1
export SGLANG_QOS_CONTROL_SOCKET=${SGLANG_QOS_CONTROL_SOCKET:-/tmp/llm-proxy-qos.sock}

exec "$project/.venv/bin/python" -m sglang.launch_server \
    --model-path "${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}" \
    --host 127.0.0.1 --port "${PORT:-8000}" \
    --disable-cuda-graph --enable-priority-scheduling "$@"
