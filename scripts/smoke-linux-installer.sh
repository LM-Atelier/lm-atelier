#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

installer="${1:-}"
version="${2:-}"
previous_installer="${3:-}"
previous_version="${4:-}"
if [[ -z "$installer" || -z "$version" ]]; then
  echo \
    "Usage: $0 <installer.run> <version> [previous-installer.run previous-version]" \
    >&2
  exit 2
fi
if [[ -n "$previous_installer" && -z "$previous_version" ]] ||
  [[ -z "$previous_installer" && -n "$previous_version" ]]; then
  echo "Previous installer and version must be provided together." >&2
  exit 2
fi
installer="$(realpath -- "$installer")"
if [[ ! -x "$installer" ]]; then
  echo "Installer is missing or not executable: $installer" >&2
  exit 1
fi
if [[ -n "$previous_installer" ]]; then
  previous_installer="$(realpath -- "$previous_installer")"
  if [[ ! -x "$previous_installer" ]]; then
    echo "Previous installer is missing or not executable: $previous_installer" >&2
    exit 1
  fi
fi

test_root="$(mktemp -d)"
trap 'rm -rf "$test_root"' EXIT
export HOME="$test_root/home"
export XDG_DATA_HOME="$test_root/data"
export LM_ATELIER_INSTALL_ROOT="$HOME/.local/opt/lm-atelier"
managed_data_root="$HOME/.local/share/lm-atelier"
mkdir -p "$HOME"

innocent_root="$HOME/innocent"
mkdir -p "$innocent_root"
printf '%s\n' "do not remove" > "$innocent_root/sentinel"
if LM_ATELIER_INSTALL_ROOT="$innocent_root" "$installer"; then
  echo "Installer replaced an unmanaged directory." >&2
  exit 1
fi
test -f "$innocent_root/sentinel"

mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications"
printf '%s\n' "user launcher" > "$HOME/.local/bin/lm-atelier"
printf '%s\n' "user desktop entry" > "$HOME/.local/share/applications/lm-atelier.desktop"
if "$installer"; then
  echo "Installer replaced user-owned launcher files." >&2
  exit 1
fi
grep -Fxq "user launcher" "$HOME/.local/bin/lm-atelier"
grep -Fxq "user desktop entry" "$HOME/.local/share/applications/lm-atelier.desktop"
rm -f "$HOME/.local/bin/lm-atelier"
rm -f "$HOME/.local/share/applications/lm-atelier.desktop"

legacy_root="$HOME/.local/opt/lm-atelier-legacy"
mkdir -p "$legacy_root/_internal"
printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "$legacy_root/lm-atelier"
printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "$legacy_root/uninstall.sh"
chmod +x "$legacy_root/lm-atelier" "$legacy_root/uninstall.sh"
printf '%s\n' '{"application": "LM Atelier"}' \
  > "$legacy_root/_internal/release-manifest.json"
ln -s "$legacy_root/lm-atelier" "$HOME/.local/bin/lm-atelier"
printf '%s\n' \
  '[Desktop Entry]' \
  'Type=Application' \
  'Name=LM Atelier' \
  'Comment=Local creative studio' \
  "Exec=\"$legacy_root/lm-atelier\"" \
  "Icon=$legacy_root/lm-atelier.png" \
  'Terminal=true' \
  'Categories=Graphics;Utility;' \
  'StartupNotify=true' \
  > "$HOME/.local/share/applications/lm-atelier.desktop"
LM_ATELIER_INSTALL_ROOT="$legacy_root" "$installer"
grep -Fxq 'lm-atelier-managed-install-v1' \
  "$legacy_root/.lm-atelier-install"
LM_ATELIER_INSTALL_ROOT="$legacy_root" "$legacy_root/uninstall.sh"
test ! -e "$legacy_root"
test ! -e "$HOME/.local/bin/lm-atelier"
test ! -e "$HOME/.local/share/applications/lm-atelier.desktop"

initial_installer="$installer"
initial_version="$version"
if [[ -n "$previous_installer" ]]; then
  initial_installer="$previous_installer"
  initial_version="$previous_version"
fi

"$initial_installer"
test -x "$LM_ATELIER_INSTALL_ROOT/lm-atelier"
test -L "$HOME/.local/bin/lm-atelier"
test -f "$HOME/.local/share/applications/lm-atelier.desktop"
.venv/bin/python scripts/inventory-frozen-payload.py \
  --payload-root "$LM_ATELIER_INSTALL_ROOT" \
  --verify-only \
  --installer-extras linux

.venv/bin/python scripts/smoke-frozen.py \
  "$LM_ATELIER_INSTALL_ROOT/lm-atelier" \
  --version "$initial_version" \
  --port 12443

mkdir -p "$managed_data_root"
printf '%s\n' "preserve" > "$managed_data_root/installer-smoke-preserve"
if [[ -n "$previous_installer" ]]; then
  "$installer"
  test -f "$managed_data_root/installer-smoke-preserve"
  .venv/bin/python scripts/inventory-frozen-payload.py \
    --payload-root "$LM_ATELIER_INSTALL_ROOT" \
    --verify-only \
    --installer-extras linux
  .venv/bin/python scripts/smoke-frozen.py \
    "$LM_ATELIER_INSTALL_ROOT/lm-atelier" \
    --version "$version" \
    --port 12443
fi
"$installer"
test -f "$managed_data_root/installer-smoke-preserve"
rm -f "$HOME/.local/bin/lm-atelier"
rm -f "$HOME/.local/share/applications/lm-atelier.desktop"
printf '%s\n' "replacement launcher" > "$HOME/.local/bin/lm-atelier"
printf '%s\n' "replacement desktop entry" > "$HOME/.local/share/applications/lm-atelier.desktop"
env -u LM_ATELIER_INSTALL_ROOT "$LM_ATELIER_INSTALL_ROOT/uninstall.sh"
test ! -e "$LM_ATELIER_INSTALL_ROOT"
grep -Fxq "replacement launcher" "$HOME/.local/bin/lm-atelier"
grep -Fxq "replacement desktop entry" "$HOME/.local/share/applications/lm-atelier.desktop"
test -f "$managed_data_root/installer-smoke-preserve"
rm -f "$HOME/.local/bin/lm-atelier"
rm -f "$HOME/.local/share/applications/lm-atelier.desktop"

"$installer"
protected_xdg_parent="$test_root/protected-xdg"
protected_local_data="$test_root/protected-local-data"
export XDG_DATA_HOME="$protected_xdg_parent/../protected-xdg"
export LOCAL_LM_DATA_DIR="$protected_local_data"
mkdir -p "$protected_xdg_parent/lm-atelier" "$protected_local_data"
printf '%s\n' "protected" > "$protected_xdg_parent/lm-atelier/sentinel"
printf '%s\n' "protected" > "$protected_local_data/sentinel"
env -u LM_ATELIER_INSTALL_ROOT \
  "$LM_ATELIER_INSTALL_ROOT/uninstall.sh" --purge-data
test ! -e "$LM_ATELIER_INSTALL_ROOT"
test ! -e "$managed_data_root"
grep -Fxq "protected" "$protected_xdg_parent/lm-atelier/sentinel"
grep -Fxq "protected" "$protected_local_data/sentinel"

echo "Linux installer smoke test passed: $installer"
