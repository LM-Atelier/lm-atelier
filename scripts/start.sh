#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ ! -d apps/web/dist ]]; then
  npm run build
fi
exec .venv/bin/lm-atelier
