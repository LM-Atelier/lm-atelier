#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else "LM Atelier source setup requires Python 3.12.")'
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './services/api[dev,package]'
npm ci
npm run build
echo "Setup complete. Run scripts/start.sh or scripts/dev.sh."
