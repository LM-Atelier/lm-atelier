from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPOSITORY = "LM-Atelier/lm-atelier"
RUNTIME_PATHS = (
    ".github/workflows/ci.yml",
    "scripts/ci-plan.py",
    "scripts/ci-merge-gate.py",
)


def configure_queue(
    tmp_path: Path,
    *,
    identity_fault: str | None = None,
    undeployed: bool = False,
    readiness_api_failure: bool = False,
    readback_fault: str | None = None,
    full_apply: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is required to exercise repository configuration")
    repository = tmp_path / "repository"
    paths = (
        "scripts/configure-public-repository.ps1",
        ".github/rulesets/public-branches.json",
        ".github/rulesets/protected-release-tags.json",
        ".github/rulesets/release-tag-creation.json",
        ".github/rulesets/public-develop-queue.json",
        *RUNTIME_PATHS,
    )
    for relative in paths:
        source = ROOT / relative
        if source.exists():
            destination = repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    fixture = {
        "repository": {
            "id": 1308948872,
            "full_name": REPOSITORY,
            "visibility": "public",
            "owner": {
                "id": 325157610,
                "type": "Organization",
            },
            "allow_squash_merge": True,
            "squash_merge_commit_title": "PR_TITLE",
            "squash_merge_commit_message": "PR_BODY",
        },
        # The readback is supplied independently of the requested POST body.
        "queue_readback": {
            "id": 1234,
            "name": "Queue verified develop changes",
            "target": "branch",
            "enforcement": "active",
            "bypass_actors": [
                {"actor_id": 32660587, "actor_type": "User", "bypass_mode": "pull_request"}
            ],
            "conditions": {"ref_name": {"include": ["refs/heads/develop"], "exclude": []}},
            "rules": [
                {
                    "type": "merge_queue",
                    "parameters": {
                        "check_response_timeout_minutes": 60,
                        "grouping_strategy": "ALLGREEN",
                        "max_entries_to_build": 3,
                        "max_entries_to_merge": 1,
                        "merge_method": "SQUASH",
                        "min_entries_to_merge": 1,
                        "min_entries_to_merge_wait_minutes": 0,
                    },
                }
            ],
        },
        "readback_fault": readback_fault,
        "readiness_api_failure": readiness_api_failure,
        "full_apply": full_apply,
        "contents": {
            path: {
                "encoding": "base64",
                "content": base64.b64encode(
                    b"old workflow" if undeployed else (ROOT / path).read_bytes()
                ).decode("ascii"),
            }
            for path in RUNTIME_PATHS
        },
    }
    if identity_fault == "owner-id":
        fixture["repository"]["owner"]["id"] = 1
    elif identity_fault == "owner-type":
        fixture["repository"]["owner"]["type"] = "User"
    elif identity_fault == "repository-id":
        fixture["repository"]["id"] = 1
    elif identity_fault == "full-name":
        fixture["repository"]["full_name"] = "LM-Atelier/another-repository"
    elif identity_fault == "full-name-case":
        fixture["repository"]["full_name"] = REPOSITORY.lower()
    if readback_fault == "incorrect-queue":
        fixture["queue_readback"]["enforcement"] = "disabled"
    elif readback_fault == "missing-rule":
        fixture["queue_readback"]["rules"] = []
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    log = tmp_path / "requests.jsonl"
    wrapper = tmp_path / "configure.ps1"
    wrapper.write_text(
        r"""
param([string]$FixturePath, [string]$LogPath, [string]$ConfigurationScript)
$ErrorActionPreference = 'Stop'
$global:QueueApiFixture = Get-Content -Raw -LiteralPath $FixturePath | ConvertFrom-Json -AsHashtable
$global:QueueCreated = $false
$global:QueueSettingsPatched = $false
function global:gh {
    $global:LASTEXITCODE = 0
    $Arguments = @($args)
    if ($Arguments[0] -eq 'auth') { return 'Fixture authentication' }
    $MethodIndex = [Array]::IndexOf($Arguments, '--method')
    $Method = $Arguments[$MethodIndex + 1]
    $InputIndex = [Array]::IndexOf($Arguments, '--input')
    $Endpoint = if ($InputIndex -ge 0) { $Arguments[$InputIndex - 1] } else { $Arguments[-1] }
    $Body = if ($InputIndex -ge 0) {
        Get-Content -Raw -LiteralPath $Arguments[$InputIndex + 1] | ConvertFrom-Json -AsHashtable
    } else { $null }
    $Record = @{method=$Method; endpoint=$Endpoint; body=$Body} | ConvertTo-Json -Depth 30 -Compress
    [IO.File]::AppendAllText($LogPath, $Record + "`n")
    $Repo = 'repos/LM-Atelier/lm-atelier'
    if ($Endpoint -eq $Repo) {
        if ($Method -eq 'PATCH') { $global:QueueSettingsPatched = $true }
        $RepositoryResponse = @{} + $global:QueueApiFixture.repository
        if ($global:QueueSettingsPatched) {
            $RepositoryResponse.squash_merge_commit_title = 'PR_TITLE'
            $RepositoryResponse.squash_merge_commit_message = 'BLANK'
            if ($global:QueueApiFixture.readback_fault -eq 'incorrect-settings') {
                $RepositoryResponse.squash_merge_commit_message = 'PR_BODY'
            }
        }
        return (ConvertTo-Json -InputObject $RepositoryResponse -Depth 30)
    }
    if ($Endpoint -eq "$Repo/branches/develop") {
        if ($global:QueueApiFixture.readiness_api_failure) {
            $global:LASTEXITCODE = 1
            return 'Fixture readiness API failure'
        }
        return '{"commit":{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}'
    }
    if ($Endpoint.StartsWith("$Repo/contents/")) {
        $Path = $Endpoint.Substring("$Repo/contents/".Length).Split('?')[0]
        return (ConvertTo-Json -InputObject $global:QueueApiFixture.contents[$Path] -Depth 30)
    }
    if ($Endpoint -eq "$Repo/rulesets") {
        if ($Method -eq 'POST') {
            $global:QueueCreated = $true
            return (ConvertTo-Json -InputObject $Body -Depth 30)
        }
        if (
            -not $global:QueueCreated -or
            $global:QueueApiFixture.readback_fault -eq 'missing-queue'
        ) { return '[]' }
        $Summary = @(@{id=1234; name='Queue verified develop changes'})
        return (ConvertTo-Json -InputObject $Summary -Depth 30)
    }
    if ($Endpoint -eq "$Repo/rulesets/1234" -and $Method -eq 'GET') {
        return (ConvertTo-Json -InputObject $global:QueueApiFixture.queue_readback -Depth 30)
    }
    if ($global:QueueApiFixture.full_apply -and $Method -in @('PATCH', 'PUT', 'POST')) {
        return '{}'
    }
    throw "Unexpected request: $Method $Endpoint"
}
if ($global:QueueApiFixture.full_apply) {
    & $ConfigurationScript -Apply
} else {
    & $ConfigurationScript -Apply -MergeQueueOnly
}
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-File",
            str(wrapper),
            "-FixturePath",
            str(fixture_path),
            "-LogPath",
            str(log),
            "-ConfigurationScript",
            str(repository / "scripts/configure-public-repository.ps1"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    requests = (
        [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if log.exists()
        else []
    )
    return result, requests


def test_queue_configuration_changes_only_queue_and_squash_settings(
    tmp_path: Path,
) -> None:
    result, requests = configure_queue(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    mutations = [request for request in requests if request["method"] != "GET"]
    assert [(item["method"], item["endpoint"]) for item in mutations] == [
        ("PATCH", f"repos/{REPOSITORY}"),
        ("POST", f"repos/{REPOSITORY}/rulesets"),
    ]
    assert mutations[0]["body"] == {
        "squash_merge_commit_title": "PR_TITLE",
        "squash_merge_commit_message": "BLANK",
    }
    queue = mutations[1]["body"]
    assert isinstance(queue, dict)
    assert queue["conditions"]["ref_name"] == {
        "include": ["refs/heads/develop"],
        "exclude": [],
    }
    assert queue["bypass_actors"] == [
        {"actor_id": 32660587, "actor_type": "User", "bypass_mode": "pull_request"}
    ]
    assert [rule["type"] for rule in queue["rules"]] == ["merge_queue"]
    parameters = queue["rules"][0]["parameters"]
    assert parameters["merge_method"] == "SQUASH"
    assert parameters["grouping_strategy"] == "ALLGREEN"
    assert parameters["max_entries_to_build"] == 3
    assert parameters["max_entries_to_merge"] == 1
    assert parameters["check_response_timeout_minutes"] >= 60
    assert "applied and verified" in result.stdout


@pytest.mark.parametrize(
    "identity_fault",
    ["owner-id", "owner-type", "repository-id", "full-name", "full-name-case"],
)
def test_queue_configuration_refuses_wrong_repository_identity(
    tmp_path: Path, identity_fault: str
) -> None:
    result, requests = configure_queue(tmp_path, identity_fault=identity_fault)
    assert result.returncode != 0
    assert "owner identity" in result.stdout + result.stderr
    assert all(request["method"] == "GET" for request in requests)


@pytest.mark.parametrize("full_apply", [False, True], ids=["queue-only", "full-apply"])
@pytest.mark.parametrize("failure", ["undeployed", "api-failure"])
def test_queue_readiness_fails_before_either_mode_mutates(
    tmp_path: Path, full_apply: bool, failure: str
) -> None:
    result, requests = configure_queue(
        tmp_path,
        full_apply=full_apply,
        undeployed=failure == "undeployed",
        readiness_api_failure=failure == "api-failure",
    )
    assert all(request["method"] == "GET" for request in requests), requests
    assert result.returncode != 0
    reason = "not deployed on develop" if failure == "undeployed" else "API request failed"
    assert reason in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("readback_fault", "reason"),
    [
        ("incorrect-queue", "Develop merge queue does not match"),
        ("missing-rule", "Develop merge queue does not match"),
        ("missing-queue", "merge queue ruleset could not be verified"),
        ("incorrect-settings", "Queue commit settings does not match"),
    ],
)
def test_queue_configuration_refuses_incorrect_or_missing_readback(
    tmp_path: Path, readback_fault: str, reason: str
) -> None:
    result, requests = configure_queue(tmp_path, readback_fault=readback_fault)
    assert any(request["method"] == "POST" for request in requests)
    assert result.returncode != 0
    assert reason in result.stdout + result.stderr
    assert "applied and verified" not in result.stdout
