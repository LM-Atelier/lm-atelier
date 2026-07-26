#!/usr/bin/env bash
set -euo pipefail

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
install_root="${LM_ATELIER_INSTALL_ROOT:-$script_root}"
home_root="$(realpath -m -- "$HOME")"
managed_data_parent="$home_root/.local/share"
data_root="$managed_data_parent/lm-atelier"
managed_marker=".lm-atelier-install"
launcher_path="$HOME/.local/bin/lm-atelier"
desktop_path="$HOME/.local/share/applications/lm-atelier.desktop"

is_managed_install() {
  local root="$1"
  if [[ -f "$root/$managed_marker" ]] &&
    grep -qx 'lm-atelier-managed-install-v1' "$root/$managed_marker"; then
    return 0
  fi
  [[ -x "$root/lm-atelier" ]] &&
    [[ -x "$root/uninstall.sh" ]] &&
    [[ -f "$root/_internal/release-manifest.json" ]] &&
    grep -q '"application": "LM Atelier"' "$root/_internal/release-manifest.json"
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
    grep -Fxq 'X-LM-Atelier-Managed=true' "$path" &&
    grep -Fxq "Exec=\"$install_root/lm-atelier\"" "$path"
}

validate_managed_data_root() {
  local resolved_parent
  local resolved_root
  if [[ -z "$home_root" || "$home_root" == "/" ]]; then
    echo "Refusing to purge data for an unsafe home path: $home_root" >&2
    return 1
  fi
  if ! resolved_parent="$(realpath -m -- "$managed_data_parent")"; then
    echo "Refusing to purge an unresolved data parent: $managed_data_parent" >&2
    return 1
  fi
  if [[ "$resolved_parent" != "$managed_data_parent" ]]; then
    echo "Refusing to purge through a redirected data parent: $managed_data_parent" >&2
    return 1
  fi
  if [[ -L "$data_root" ]]; then
    echo "Refusing to purge a symbolic-link data root: $data_root" >&2
    return 1
  fi
  if ! resolved_root="$(realpath -m -- "$data_root")"; then
    echo "Refusing to purge an unresolved data root: $data_root" >&2
    return 1
  fi
  if [[ "$resolved_root" != "$managed_data_parent/lm-atelier" ]] ||
    [[ "$(dirname -- "$resolved_root")" != "$managed_data_parent" ]] ||
    [[ "$(basename -- "$resolved_root")" != "lm-atelier" ]]; then
    echo "Refusing to purge an unexpected data root: $resolved_root" >&2
    return 1
  fi
}

install_root="$(realpath -m -- "$install_root")"
case "$install_root" in
  "$HOME"/*) ;;
  *)
    echo "Refusing to remove an unsafe installation path: $install_root" >&2
    exit 1
    ;;
esac
if ! is_managed_install "$install_root"; then
  echo "Refusing to remove a directory not managed by LM Atelier: $install_root" >&2
  exit 1
fi
if [[ "${1:-}" == "--purge-data" ]]; then
  validate_managed_data_root
fi

if launcher_points_to_install "$launcher_path"; then
  rm -f -- "$launcher_path"
fi
if is_managed_desktop_entry "$desktop_path"; then
  rm -f -- "$desktop_path"
fi
rm -rf -- "$install_root"

if [[ "${1:-}" == "--purge-data" ]]; then
  rm -rf -- "$data_root"
  echo "LM Atelier and its local data were removed."
else
  echo "LM Atelier was removed. Local data was not removed."
fi
