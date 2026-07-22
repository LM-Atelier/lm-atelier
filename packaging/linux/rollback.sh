#!/usr/bin/env bash
set -euo pipefail

install_root="${LM_ATELIER_INSTALL_ROOT:-$HOME/.local/share/lm-atelier}"
if [[ ! -L "$install_root/previous" ]]; then
  echo "No previous LM Atelier version is available." >&2
  exit 1
fi
current="$(readlink "$install_root/current")"
previous="$(readlink "$install_root/previous")"
ln -sfn "$previous" "$install_root/current"
ln -sfn "$current" "$install_root/previous"
echo "Rolled back to $(basename "$previous"). User data was unchanged."
