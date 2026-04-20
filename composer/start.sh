#!/usr/bin/env bash
# bash required: `wait -n` is not POSIX (dash/sh errors with "Illegal option -n").
set -euo pipefail

# Allow NVIDIA_NIM_API_KEY as an alias for NVIDIA_API_KEY.
if [ -n "${NVIDIA_NIM_API_KEY:-}" ] && [ -z "${NVIDIA_API_KEY:-}" ]; then
  export NVIDIA_API_KEY="${NVIDIA_NIM_API_KEY}"
fi

# Internal/backend URL used by the Gradio frontend.
export BACKEND_BASE_URL="${BACKEND_BASE_URL:-http://127.0.0.1:9012}"
export BACKEND_PORT="${BACKEND_PORT:-9012}"
export PORT="${PORT:-7860}"

python /app/backend_server.py &
backend_pid=$!

python /app/frontend/frontend_server.py &
frontend_pid=$!

cleanup() {
  kill "$backend_pid" "$frontend_pid" 2>/dev/null || true
}
trap cleanup INT TERM

wait -n "$backend_pid" "$frontend_pid"
cleanup
