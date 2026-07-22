#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

.venv/bin/ruff format --check services/api
.venv/bin/ruff check services/api
.venv/bin/mypy services/api/local_lm
.venv/bin/pytest services/api/tests -q
npm run lint
npm run typecheck
npm test
npm run build
