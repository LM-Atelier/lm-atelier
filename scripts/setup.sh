#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './services/api[dev]'
npm ci
npm run build
echo "Setup complete. Run scripts/start.sh or scripts/dev.sh."
