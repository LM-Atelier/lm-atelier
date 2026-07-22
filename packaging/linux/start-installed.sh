#!/usr/bin/env bash
set -euo pipefail

install_root="${LM_ATELIER_INSTALL_ROOT:-$HOME/.local/share/lm-atelier}"
export LOCAL_LM_DATA_DIR="${LOCAL_LM_DATA_DIR:-$install_root/data}"
export LOCAL_LM_HOST="${LOCAL_LM_HOST:-127.0.0.1}"
cd "$install_root/current"
exec .venv/bin/lm-atelier
