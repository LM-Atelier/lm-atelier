#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

version="$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' services/api/local_lm/__init__.py)"
if [[ -z "$version" ]]; then
  echo "Could not determine the LM Atelier version." >&2
  exit 1
fi

output_root="${1:-release}"
icon_root="build/installer-assets"
dist_root="build/pyinstaller-linux"
work_root="build/pyinstaller-work-linux"
staging_root="$(mktemp -d)"
payload_path="$staging_root/payload.tar.gz"
header_path="$staging_root/installer.sh"
installer="$output_root/LM-Atelier-Setup-$version-linux-x86_64.run"
trap 'rm -rf "$staging_root"' EXIT

npm run build
.venv/bin/python scripts/build-icons.py --output-dir "$icon_root"
.venv/bin/python -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "$dist_root" \
  --workpath "$work_root" \
  packaging/LMAtelier.spec
.venv/bin/python scripts/smoke-frozen.py \
  "$dist_root/lm-atelier/lm-atelier" \
  --version "$version"

cp -R "$dist_root/lm-atelier/." "$staging_root/"
install -m 755 packaging/linux/frozen-uninstall.sh "$staging_root/uninstall.sh"
install -m 644 "$icon_root/lm-atelier.png" "$staging_root/lm-atelier.png"
tar -C "$staging_root" \
  --exclude="payload.tar.gz" \
  --exclude="installer.sh" \
  --sort=name \
  --mtime="@0" \
  --owner=0 \
  --group=0 \
  -czf "$payload_path" .
sed "s/@VERSION@/$version/g" packaging/linux/self-extracting-installer.sh > "$header_path"

mkdir -p "$output_root"
cp "$header_path" "$installer"
cat "$payload_path" >> "$installer"
chmod 755 "$installer"
sha256sum "$installer"
echo "Created $installer"
