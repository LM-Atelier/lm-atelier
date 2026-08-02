#!/usr/bin/env bash
set -euo pipefail

version="@VERSION@"
install_root="${LM_ATELIER_INSTALL_ROOT:-$HOME/.local/opt/lm-atelier}"
data_root="${XDG_DATA_HOME:-$HOME/.local/share}/lm-atelier"
launch=false
managed_marker=".lm-atelier-install"
launcher_path="$HOME/.local/bin/lm-atelier"
desktop_path="$HOME/.local/share/applications/lm-atelier.desktop"

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

is_managed_install() {
  local root="$1"
  if [[ -f "$root/$managed_marker" ]] &&
    grep -qx 'lm-atelier-managed-install-v1' "$root/$managed_marker"; then
    return 0
  fi
  [[ -x "$root/lm-atelier" ]] || return 1
  [[ -x "$root/uninstall.sh" ]] || return 1
  if [[ -f "$root/_internal/release-manifest.json" ]]; then
    grep -q '"application": "LM Atelier"' "$root/_internal/release-manifest.json"
  else
    # v0.1.x payloads predate the bundled release manifest; identify them
    # by the application's own migration package instead.
    [[ -d "$root/_internal/local_lm/migrations" ]]
  fi
}

launcher_points_to_install() {
  local path="$1"
  local target
  [[ -L "$path" ]] || return 1
  target="$(readlink -- "$path")"
  if [[ "$target" != /* ]]; then
    target="$(dirname "$path")/$target"
  fi
  [[ "$(realpath -m -- "$target")" == "$(realpath -m -- "$install_root/lm-atelier")" ]]
}

is_managed_desktop_entry() {
  local path="$1"
  [[ -f "$path" ]] &&
    [[ ! -L "$path" ]] &&
    [[ "$(grep -c '^Exec=' "$path")" -eq 1 ]] &&
    grep -Fxq '[Desktop Entry]' "$path" &&
    grep -Fxq 'Type=Application' "$path" &&
    grep -Fxq 'Name=LM Atelier' "$path" &&
    grep -Fxq "Exec=\"$install_root/lm-atelier\"" "$path" &&
    grep -Fxq "Icon=$install_root/lm-atelier.png" "$path" &&
    {
      grep -Fxq 'X-LM-Atelier-Managed=true' "$path" ||
        {
          # v0.1.x wrote this exact entry before adding an ownership marker.
          grep -Fxq 'Comment=Local creative studio' "$path" &&
            grep -Fxq 'Terminal=true' "$path" &&
            grep -Fxq 'Categories=Graphics;Utility;' "$path" &&
            grep -Fxq 'StartupNotify=true' "$path"
        }
    }
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
if [[ -e "$install_root" ]] && ! is_managed_install "$install_root"; then
  echo "Refusing to replace a directory not managed by LM Atelier: $install_root" >&2
  exit 1
fi
if [[ -e "$launcher_path" || -L "$launcher_path" ]] &&
  ! launcher_points_to_install "$launcher_path"; then
  echo "Refusing to replace a launcher not managed by this LM Atelier installation: $launcher_path" >&2
  exit 1
fi
if [[ -e "$desktop_path" ]] && ! is_managed_desktop_entry "$desktop_path"; then
  echo "Refusing to replace a desktop entry not managed by this LM Atelier installation: $desktop_path" >&2
  exit 1
fi

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
test -f "$partial_root/$managed_marker"

if [[ -e "$install_root" ]]; then
  mv "$install_root" "$backup_root"
fi
mv "$partial_root" "$install_root"
rm -rf "$backup_root"

mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications"
ln -sfn "$install_root/lm-atelier" "$launcher_path"
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
  'X-LM-Atelier-Managed=true' \
  > "$desktop_path"

echo "LM Atelier $version installed at $install_root"
echo "Run $launcher_path or open LM Atelier from your application menu."
echo "Linux image/video require an externally configured compatible media engine and are not certified."
if [[ "$launch" == true ]]; then
  # First launch after install lands in setup, mirroring the Windows installer.
  exec "$install_root/lm-atelier" --first-run-setup
fi
exit 0

__LM_ATELIER_PAYLOAD_BELOW__
