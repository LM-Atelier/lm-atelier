#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ ! -x .venv/bin/lm-atelier ]]; then
  echo "Run scripts/setup.sh first." >&2
  exit 1
fi

LOCAL_LM_DEV=true .venv/bin/lm-atelier &
api_pid=$!
trap 'kill "$api_pid" 2>/dev/null || true' EXIT INT TERM
npm run dev:web
