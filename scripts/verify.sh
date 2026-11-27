#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python_tools="$root/.venv/bin"

# With --api-tests-only the gate runs its API test stage and nothing else. CI
# runs one part of the API suite that way, in a job beside the one that runs
# the whole gate with the other part.
api_tests_only=false
if (($# == 1)) && [[ "$1" == "--api-tests-only" ]]; then
  api_tests_only=true
elif (($# > 0)); then
  echo "Usage: scripts/verify.sh [--api-tests-only]" >&2
  exit 2
fi

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

# CI runs the API suite in two parts at once, and LOCAL_LM_API_TEST_PART names
# the part a run takes, as in 1/2; without it the whole suite runs. The part's
# modules come from api-test-part.py, which lists them the way pytest collects
# the suite and refuses a part with none of them, since that part would pass
# without running a test.
run_api_tests() {
  local modules=(services/api/tests)
  if [[ -n "${LOCAL_LM_API_TEST_PART+set}" ]]; then
    local listed
    listed="$("$python_tools/python" scripts/api-test-part.py "$LOCAL_LM_API_TEST_PART")"
    if [[ -z "$listed" ]]; then
      echo "API test part $LOCAL_LM_API_TEST_PART lists no modules." >&2
      exit 1
    fi
    mapfile -t modules <<< "$listed"
  fi
  run_checked "API tests" \
    "$python_tools/pytest" "${modules[@]}" -q --durations=20 \
      "--basetemp=$pytest_temp" -p no:cacheprovider -n auto --dist loadfile
}

require_tool "$python_tools/python" "Python"
require_tool "$python_tools/pytest" "pytest"

mkdir -p "$root/temp"
pytest_temp="$root/temp/verify-pytest-$$"
# Away from the checkout unless the caller said otherwise, because the suite
# opens a data directory and the default is relative, so it would land in the
# tree. Beside the pytest scratch rather than inside it: pytest empties its
# own basetemp as it starts, and by then the application has taken ownership
# of the data directory, so one inside the other cannot both survive.
export LOCAL_LM_DATA_DIR="${LOCAL_LM_DATA_DIR:-$root/temp/verify-data-$$}"

if [[ "$api_tests_only" == true ]]; then
  run_api_tests
  echo "The API test stage passed. Run scripts/verify.sh without --api-tests-only for the whole gate."
  exit 0
fi

require_tool "$python_tools/ruff" "Ruff"
require_tool "$python_tools/mypy" "mypy"
require_tool "$python_tools/bandit" "Bandit"
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
  "$python_tools/ruff" format --check services/api scripts packaging
run_checked "Ruff lint" \
  "$python_tools/ruff" check services/api scripts packaging
# The scripts are named here for the same reason they are named in the ruff
# lines above: two of them decide whether verification happens at all, and until
# they were added the tools that judge the gate were the only Python in the
# repository the gate never judged.
run_checked "Strict mypy" \
  env MYPYPATH="$root/services/api:$root/services/api/tests" \
  "$python_tools/mypy" --explicit-package-bases --config-file services/api/pyproject.toml \
    services/api/local_lm services/api/tests scripts
run_checked "Bandit high-severity scan" \
  "$python_tools/bandit" -q -lll -r services/api/local_lm
run_checked "Version metadata" \
  "$python_tools/python" scripts/sync-version.py

run_api_tests
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
# Named because this gate discovers and typechecks the browser suite without
# running it, which reads as coverage.
echo "Not run here: the browser golden path. Use 'npm run e2e'; CI runs it on Ubuntu."
