#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python_tools="$root/.venv/bin"

require_tool() {
  local path="$1"
  local description="$2"
  if [[ ! -x "$path" ]]; then
    echo "Missing $path. Create .venv and install services/api[dev,package] first ($description)." >&2
    exit 1
  fi
}

run_checked() {
  local label="$1"
  shift
  echo "==> $label"
  "$@"
}

require_tool "$python_tools/python" "Python"
require_tool "$python_tools/ruff" "Ruff"
require_tool "$python_tools/mypy" "mypy"
require_tool "$python_tools/bandit" "Bandit"
require_tool "$python_tools/pytest" "pytest"
command -v npm >/dev/null 2>&1 || {
  echo "npm is required for web verification." >&2
  exit 1
}
command -v git >/dev/null 2>&1 || {
  echo "git is required for repository verification." >&2
  exit 1
}
command -v pwsh >/dev/null 2>&1 || {
  echo "PowerShell 7 is required to validate Windows packaging scripts." >&2
  exit 1
}

run_checked "Ruff format" \
  "$python_tools/ruff" format --check services/api
run_checked "Ruff lint" \
  "$python_tools/ruff" check services/api
run_checked "Strict mypy" \
  "$python_tools/mypy" --config-file services/api/pyproject.toml services/api/local_lm
run_checked "Bandit high-severity scan" \
  "$python_tools/bandit" -q -lll -r services/api/local_lm
run_checked "Version metadata" \
  "$python_tools/python" scripts/sync-version.py

mkdir -p "$root/temp"
pytest_temp="$root/temp/verify-pytest-$$"
run_checked "API tests" \
  "$python_tools/pytest" services/api/tests -q \
    "--basetemp=$pytest_temp" -p no:cacheprovider
run_checked "Web lint" npm run lint
run_checked "Web typecheck" npm run typecheck
run_checked "Web tests" npm test
run_checked "Browser suite typecheck" npm run e2e:typecheck
run_checked "Browser suite discovery" npm run e2e:list
run_checked "Production web build" npm run build

run_checked "GitHub workflow policy" \
  "$python_tools/python" scripts/validate-workflows.py

echo "==> Windows packaging syntax"
pwsh -NoProfile -NonInteractive -Command '
  $ErrorActionPreference = "Stop"
  $failures = @()
  Get-ChildItem -LiteralPath "packaging/windows", "scripts" -Filter "*.ps1" |
    ForEach-Object {
      $tokens = $null
      $errors = $null
      [System.Management.Automation.Language.Parser]::ParseFile(
        $_.FullName,
        [ref]$tokens,
        [ref]$errors
      ) | Out-Null
      if ($errors.Count -gt 0) {
        $failures += "$($_.Name): $($errors -join "; ")"
      }
    }
  if ($failures.Count -gt 0) {
    throw "Windows packaging syntax failed: $($failures -join " | ")"
  }
'

run_checked "Linux packaging syntax" bash -n \
  scripts/build-linux-installer.sh \
  scripts/smoke-linux-installer.sh \
  packaging/linux/self-extracting-installer.sh \
  packaging/linux/frozen-uninstall.sh

run_checked "Repository hygiene" \
  "$python_tools/python" scripts/check-repository-hygiene.py

run_checked "Unstaged whitespace check" git diff --check --
run_checked "Staged whitespace check" git diff --cached --check --
if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
  run_checked "Pull-request whitespace check" \
    git diff --check "origin/$GITHUB_BASE_REF...HEAD" --
else
  run_checked "Latest commit whitespace check" \
    git log --check --format= -1 HEAD --
fi
echo "All LM Atelier local verification gates passed."
