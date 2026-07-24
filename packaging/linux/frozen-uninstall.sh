#!/usr/bin/env bash
set -euo pipefail

install_root="${LM_ATELIER_INSTALL_ROOT:-$HOME/.local/opt/lm-atelier}"
data_root="${XDG_DATA_HOME:-$HOME/.local/share}/lm-atelier"

install_root="$(realpath -m -- "$install_root")"
data_root="$(realpath -m -- "$data_root")"
case "$install_root" in
  "$HOME"/*) ;;
  *)
    echo "Refusing to remove an unsafe installation path: $install_root" >&2
    exit 1
    ;;
esac

rm -f "$HOME/.local/bin/lm-atelier"
rm -f "$HOME/.local/share/applications/lm-atelier.desktop"
rm -rf "$install_root"

if [[ "${1:-}" == "--purge-data" ]]; then
  case "$data_root" in
    ""|"/"|"$HOME")
      echo "Refusing to remove an unsafe data path: $data_root" >&2
      exit 1
      ;;
  esac
  rm -rf "$data_root"
  echo "LM Atelier and its local data were removed."
else
  echo "LM Atelier was removed. Local data remains at $data_root"
fi
