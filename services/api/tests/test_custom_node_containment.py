"""The custom-node canary tells the truth when the host is not confined."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from local_lm.custom_node_containment import (
    ContainmentLevel,
    ContainmentProbe,
    child_environment,
    classify_observations,
    execution_authorized,
    host_containment_capability,
    main,
    offline_badge_allowed,
    probe_worker_host,
)

_RESULT_FIELDS = {
    "environment_limited",
    "read_input",
    "read_outside",
    "wrote_outside",
    "wrote_scratch",
}


def test_an_unconfined_host_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_ATELIER_CANARY_SENTINEL", "invented-sentinel")
    capability = host_containment_capability()
    probe = probe_worker_host(tmp_path)

    assert capability.backend == "none"
    assert capability.backend_version == "0"
    assert capability.profile_sha256 is None
    assert capability.file_denial_provable is False
    assert capability.connect_denial_provable is False
    assert probe.level is ContainmentLevel.UNAVAILABLE
    assert probe.platform == capability.platform
    assert probe.profile_version == 1
    assert probe.profile_sha256 is None
    assert probe.tested_at is not None
    assert probe.read_input_allowed is True
    assert probe.write_scratch_allowed is True
    assert probe.read_outside_denied is False
    assert probe.write_outside_denied is False
    assert probe.connect_denied is None
    assert probe.environment_limited is True
    assert probe.diagnostic == "ok"
    assert probe.authorizes_execution is False
    assert probe.offline_badge is False
    assert (tmp_path / "outside" / "marker.bin").read_bytes() == b"neutral-canary"
    assert (tmp_path / "outside" / "canary-outside-write").read_bytes() == b"wrote"
    assert (tmp_path / "scratch" / "input.bin").read_bytes() == b"neutral-input"
    assert (tmp_path / "scratch" / "canary-scratch-result").read_bytes() == b"ok"
    assert b"invented-sentinel" not in (tmp_path / "outside" / "canary-outside-write").read_bytes()


def test_the_canary_prints_only_the_bounded_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = tmp_path / "marker.bin"
    marker.write_bytes(b"neutral-canary")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "input.bin").write_bytes(b"neutral-input")

    code = main(["--canary", str(scratch), str(marker)])
    captured = capsys.readouterr().out

    assert code == 0
    payload = json.loads(captured)
    assert set(payload) == _RESULT_FIELDS
    assert "neutral-canary" not in captured
    assert "neutral-input" not in captured


def test_a_missing_fixture_is_not_a_denial(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    marker = tmp_path / "marker.bin"

    assert main(["--canary", str(scratch), str(marker)]) == 3


def test_a_canary_that_does_not_finish_is_not_a_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=2, stdout="")

    monkeypatch.setattr("local_lm.custom_node_containment.subprocess.run", fail_run)
    probe = probe_worker_host(tmp_path)

    assert probe.level is ContainmentLevel.UNAVAILABLE
    assert probe.diagnostic == "canary-no-result"
    assert probe.read_outside_denied is None
    assert probe.write_outside_denied is None
    assert probe.tested_at is None
    assert probe.authorizes_execution is False
    assert not (tmp_path / "outside" / "canary-outside-write").exists()


def test_a_launch_failure_has_a_fixed_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError(1, "launch-detail")

    monkeypatch.setattr("local_lm.custom_node_containment.subprocess.run", fail_run)
    probe = probe_worker_host(tmp_path)

    assert probe.diagnostic == "canary-launch-failed"
    assert probe.read_outside_denied is None
    assert "launch-detail" not in probe.diagnostic


def test_a_timeout_is_not_a_denial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="canary", timeout=30)

    monkeypatch.setattr("local_lm.custom_node_containment.subprocess.run", fail_run)
    probe = probe_worker_host(tmp_path)

    assert probe.diagnostic == "canary-timeout"
    assert probe.read_outside_denied is None
    assert probe.write_outside_denied is None


def test_malformed_output_is_not_a_denial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"read_outside": false, "extra": true}',
        )

    monkeypatch.setattr("local_lm.custom_node_containment.subprocess.run", fail_run)
    probe = probe_worker_host(tmp_path)

    assert probe.diagnostic == "canary-malformed-output"
    assert probe.read_outside_denied is None
    assert probe.write_outside_denied is None


def test_the_child_environment_omits_an_invented_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_ATELIER_CANARY_SENTINEL", "invented-sentinel")
    env = child_environment(r"C:\neutral\api")

    assert "LM_ATELIER_CANARY_SENTINEL" not in env
    assert "invented-sentinel" not in env.values()
    assert env["LC_CTYPE"] == "C.UTF-8"
    assert env["PYTHONPATH"] == r"C:\neutral\api"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_file_denial_without_a_refused_connect_is_not_verified() -> None:
    level = classify_observations(
        read_outside_denied=True,
        write_outside_denied=True,
        connect_denied=False,
        capabilities_held=True,
    )

    assert level is ContainmentLevel.FILESYSTEM_RESTRICTED


def test_observations_do_not_mint_a_verified_level() -> None:
    no_egress = classify_observations(
        read_outside_denied=True,
        write_outside_denied=True,
        connect_denied=True,
        capabilities_held=True,
    )
    process_tree = classify_observations(
        read_outside_denied=False,
        write_outside_denied=False,
        connect_denied=False,
        capabilities_held=True,
        process_tree_limited=True,
    )
    blocked_worker = classify_observations(
        read_outside_denied=True,
        write_outside_denied=True,
        connect_denied=True,
        capabilities_held=False,
    )

    assert no_egress is ContainmentLevel.FILESYSTEM_NO_EGRESS
    assert process_tree is ContainmentLevel.PROCESS_TREE_ONLY
    assert blocked_worker is ContainmentLevel.UNAVAILABLE


def test_invalid_incomplete_or_stale_evidence_cannot_authorize() -> None:
    now = datetime.now(UTC)
    fresh_after = now - timedelta(hours=1)
    incomplete = _probe(profile_sha256=None, tested_at=None, level=ContainmentLevel.UNAVAILABLE)
    stale = _probe(
        profile_sha256="a" * 64,
        tested_at="2020-01-01T00:00:00+00:00",
        level=ContainmentLevel.VERIFIED,
    )
    naive_time = _probe(
        profile_sha256="b" * 64,
        tested_at="2026-10-02T00:00:00",
        level=ContainmentLevel.VERIFIED,
    )
    foreign_backend = _probe(
        profile_sha256="c" * 64,
        tested_at=now.isoformat(),
        level=ContainmentLevel.VERIFIED,
        backend="test-backend",
    )
    constructed = _probe(
        profile_sha256="d" * 64,
        tested_at=now.isoformat(),
        level=ContainmentLevel.VERIFIED,
        offline_badge=True,
        authorizes_execution=True,
    )

    probes = (incomplete, stale, naive_time, foreign_backend, constructed)
    assert all(execution_authorized(probe, fresh_after=fresh_after) is False for probe in probes)
    assert all(offline_badge_allowed(probe, fresh_after=fresh_after) is False for probe in probes)
    assert execution_authorized(constructed, fresh_after=None) is False


def _probe(
    *,
    profile_sha256: str | None,
    tested_at: str | None,
    level: ContainmentLevel,
    backend: str = "none",
    offline_badge: bool = False,
    authorizes_execution: bool = False,
) -> ContainmentProbe:
    return ContainmentProbe(
        level=level,
        platform="test",
        profile_version=1,
        backend=backend,
        backend_version="0",
        profile_sha256=profile_sha256,
        tested_at=tested_at,
        read_input_allowed=None,
        write_scratch_allowed=None,
        read_outside_denied=None,
        write_outside_denied=None,
        connect_denied=None,
        environment_limited=None,
        diagnostic="ok",
        authorizes_execution=authorizes_execution,
        offline_badge=offline_badge,
    )
