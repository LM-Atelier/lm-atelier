#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ "${1:-}" != "--skip-verify" ]]; then
  ./scripts/verify.sh
fi

version="$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' services/api/local_lm/__init__.py)"
if [[ -z "$version" ]]; then
  echo "Could not determine LM Atelier version." >&2
  exit 1
fi

npm run build
mkdir -p release
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
bundle="$staging/lm-atelier-$version"
mkdir -p "$bundle"
git archive HEAD | tar -x -C "$bundle"
mkdir -p "$bundle/apps/web"
cp -R apps/web/dist "$bundle/apps/web/dist"

tar -C "$staging" -czf "release/lm-atelier-$version.tar.gz" "lm-atelier-$version"
(
  cd release
  sha256sum "lm-atelier-$version.tar.gz" > SHA256SUMS
)
echo "Created release/lm-atelier-$version.tar.gz"
echo "Build the Windows ZIP with scripts/package.ps1 on Windows."
