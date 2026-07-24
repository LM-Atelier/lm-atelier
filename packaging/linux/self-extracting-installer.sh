#!/usr/bin/env bash
set -euo pipefail

version="@VERSION@"
install_root="${LM_ATELIER_INSTALL_ROOT:-$HOME/.local/opt/lm-atelier}"
data_root="${XDG_DATA_HOME:-$HOME/.local/share}/lm-atelier"
launch=false

usage() {
  cat <<EOF
LM Atelier $version installer

Usage: $0 [--launch] [--uninstall] [--purge-data]

Installs the application to:
  $install_root

Application data is stored separately at:
  $data_root
EOF
}

for argument in "$@"; do
  case "$argument" in
    --launch) launch=true ;;
    --uninstall)
      if [[ -x "$install_root/uninstall.sh" ]]; then
        exec "$install_root/uninstall.sh"
      fi
      echo "LM Atelier is not installed at $install_root" >&2
      exit 1
      ;;
    --purge-data)
      if [[ -x "$install_root/uninstall.sh" ]]; then
        exec "$install_root/uninstall.sh" --purge-data
      fi
      echo "LM Atelier is not installed at $install_root" >&2
      exit 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $argument" >&2
      usage >&2
      exit 2
      ;;
  esac
done

install_root="$(realpath -m -- "$install_root")"
case "$install_root" in
  "$HOME"/*) ;;
  *)
    echo "Refusing to install to an unsafe path: $install_root" >&2
    exit 1
    ;;
esac

payload_line="$(awk '/^__LM_ATELIER_PAYLOAD_BELOW__$/ { print NR + 1; exit }' "$0")"
if [[ -z "$payload_line" ]]; then
  echo "The installer payload is missing." >&2
  exit 1
fi

parent_root="$(dirname "$install_root")"
partial_root="${install_root}.partial.$$"
backup_root="${install_root}.previous.$$"
mkdir -p "$parent_root"
rm -rf "$partial_root" "$backup_root"
mkdir -p "$partial_root"
cleanup() {
  rm -rf "$partial_root"
  if [[ -e "$backup_root" && ! -e "$install_root" ]]; then
    mv "$backup_root" "$install_root"
  else
    rm -rf "$backup_root"
  fi
}
trap cleanup EXIT

tail -n +"$payload_line" "$0" | tar -xzf - -C "$partial_root"
test -x "$partial_root/lm-atelier"
test -x "$partial_root/uninstall.sh"

if [[ -e "$install_root" ]]; then
  mv "$install_root" "$backup_root"
fi
mv "$partial_root" "$install_root"
rm -rf "$backup_root"

mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications"
ln -sfn "$install_root/lm-atelier" "$HOME/.local/bin/lm-atelier"
printf '%s\n' \
  '[Desktop Entry]' \
  'Type=Application' \
  'Name=LM Atelier' \
  'Comment=Local creative studio' \
  "Exec=\"$install_root/lm-atelier\"" \
  "Icon=$install_root/lm-atelier.png" \
  'Terminal=true' \
  'Categories=Graphics;Utility;' \
  'StartupNotify=true' \
  > "$HOME/.local/share/applications/lm-atelier.desktop"

echo "LM Atelier $version installed at $install_root"
echo "Run $HOME/.local/bin/lm-atelier or open LM Atelier from your application menu."
if [[ "$launch" == true ]]; then
  exec "$install_root/lm-atelier"
fi
exit 0

__LM_ATELIER_PAYLOAD_BELOW__
