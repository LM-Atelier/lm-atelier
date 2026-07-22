#!/usr/bin/env bash
set -euo pipefail

install_root="${LM_ATELIER_INSTALL_ROOT:-$HOME/.local/share/lm-atelier}"
rm -f "$HOME/.local/bin/lm-atelier"
rm -rf "$install_root/current" "$install_root/previous" "$install_root/versions"
if [[ "${1:-}" == "--purge-data" ]]; then
  rm -rf "$install_root/data"
  echo "LM Atelier and its local data were removed."
else
  echo "LM Atelier was removed. Local data remains at $install_root/data"
fi
