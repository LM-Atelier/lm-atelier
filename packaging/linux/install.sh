#!/usr/bin/env bash
set -euo pipefail

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
version="$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "$source_root/services/api/local_lm/__init__.py")"
install_root="${LM_ATELIER_INSTALL_ROOT:-$HOME/.local/share/lm-atelier}"
version_root="$install_root/versions/$version"

if [[ ! -d "$source_root/apps/web/dist" ]]; then
  echo "This archive has no prebuilt web application. Use an official release archive." >&2
  exit 1
fi

mkdir -p "$install_root/versions" "$install_root/data"
rm -rf "$version_root.partial"
mkdir -p "$version_root.partial"
cp -R "$source_root"/. "$version_root.partial/"
python3 -m venv "$version_root.partial/.venv"
"$version_root.partial/.venv/bin/python" -m pip install --upgrade pip
"$version_root.partial/.venv/bin/python" -m pip install "$version_root.partial/services/api"
rm -rf "$version_root"
mv "$version_root.partial" "$version_root"

if [[ -L "$install_root/current" ]]; then
  previous="$(readlink "$install_root/current")"
  ln -sfn "$previous" "$install_root/previous"
fi
ln -sfn "$version_root" "$install_root/current"
mkdir -p "$HOME/.local/bin"
ln -sfn "$install_root/current/packaging/linux/start-installed.sh" "$HOME/.local/bin/lm-atelier"
echo "LM Atelier $version installed. Run $HOME/.local/bin/lm-atelier"
