"""The Registry trust policy: what installs without asking, and what never does.

Two layers are pinned separately. The decision is pure and is tested against
every kind of resolution it can meet. The recorder is tested through the same
verification an explicit review runs, with only the launch-contract read stubbed
the way the neighbouring activation tests stub it.

The cases that matter most are the ones where trust must NOT appear: a git
source, a security refusal, a resolution describing a different package, bytes
that differ from the ones verified, and above all a package a person already
refused - including a refusal recorded before the authority field existed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import local_lm.comfy_registry_activation as activation_module
from local_lm.comfy_registry import ComfyNodeResolution
from local_lm.comfy_registry_activation import (
    ComfyRegistryActivationError,
    record_registry_policy_trust,
    review_comfy_registry_install,
)
from local_lm.comfy_registry_installs import ComfyRegistryLaunchContract
from local_lm.db import Base
from local_lm.models import ComfyRegistryInstall
from local_lm.registry_trust_policy import POLICY_ID, decide_registry_trust

_ARCHIVE = "a" * 64
_MANIFEST = hashlib.sha256(b"").hexdigest()


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


@pytest.fixture(autouse=True)
def registry_source_folder(tmp_path: Path) -> None:
    (tmp_path / "lm-atelier-registry_example").mkdir()


def _install(
    session: Session,
    *,
    trusted: bool = False,
    review: dict[str, Any] | None = None,
) -> ComfyRegistryInstall:
    install = ComfyRegistryInstall(
        package_id="comfyui-example-node",
        package_version="1.2.3",
        registry_record_id="record-123",
        repository_url="https://github.com/example/comfyui-example-node.git",
        download_url="https://cdn.comfy.org/example/1.2.3.zip",
        archive_sha256=_ARCHIVE,
        manifest_sha256=_MANIFEST,
        installed_path="lm-atelier-registry_example",
        node_types_json=["ExampleNode"],
        pip_dependencies_json=[],
        review_json=review if review is not None else {"review_required": True},
        wheel_closure_sha256="c" * 64,
        wheel_environment_sha256="d" * 64,
        wheel_environment_path=f"registry-wheels-v3-{'c' * 64}",
        trusted=trusted,
        active=False,
    )
    session.add(install)
    session.commit()
    return install


def _resolution(**updates: Any) -> ComfyNodeResolution:
    fields: dict[str, Any] = {
        "package_id": "comfyui-example-node",
        "declared_version": "1.2.3",
        "node_types": ("ExampleNode",),
        "install_kind": "registry_archive",
        "repository_url": "https://github.com/example/comfyui-example-node.git",
        "registry_record_id": "record-123",
        "download_url": "https://cdn.comfy.org/example/1.2.3.zip",
    }
    fields.update(updates)
    return ComfyNodeResolution(**fields)


# ---- the decision -------------------------------------------------------------


def test_a_registry_archive_that_passed_the_existing_review_is_trusted_without_asking(
    session: Session,
) -> None:
    decision = decide_registry_trust(_resolution(), _install(session))

    assert decision.outcome == "auto_trust"
    assert decision.reason == POLICY_ID
    assert decision.notices == ()


def test_a_deprecated_version_is_still_trusted_and_the_deprecation_is_shown(
    session: Session,
) -> None:
    """Deprecation is not a security finding, and hiding it would not be honest either."""

    decision = decide_registry_trust(
        _resolution(warnings=("deprecated_version",)), _install(session)
    )

    assert decision.outcome == "auto_trust"
    assert decision.notices == ("deprecated_version",)


def test_a_git_source_pauses_once_in_plain_language(session: Session) -> None:
    decision = decide_registry_trust(
        _resolution(install_kind="git_commit", warnings=("source_review_required",)),
        _install(session),
    )

    assert decision.outcome == "pause"
    assert decision.reason == "source_review_required"
    assert "code repository" in decision.explanation
    assert "_" not in decision.explanation


@pytest.mark.parametrize(
    "error_code",
    ["registry_security_warning", "registry_package_inactive", "registry_version_inactive"],
)
def test_every_existing_refusal_stays_a_refusal(session: Session, error_code: str) -> None:
    decision = decide_registry_trust(_resolution(error_code=error_code), _install(session))

    assert decision.outcome == "refuse"
    assert decision.reason == error_code


def test_a_warning_nobody_has_seen_pauses_rather_than_trusting(session: Session) -> None:
    """A trust boundary that fails open on an unknown signal is not a boundary."""

    decision = decide_registry_trust(
        _resolution(warnings=("some_future_warning",)), _install(session)
    )

    assert decision.outcome == "pause"
    assert decision.reason == "unrecognised_warning"


@pytest.mark.parametrize(
    "different",
    [
        {"package_id": "comfyui-other-node"},
        {"declared_version": "1.2.4"},
        {"registry_record_id": "record-999"},
        {"download_url": "https://cdn.comfy.org/example/other.zip"},
        {"repository_url": "https://github.com/someone-else/fork.git"},
    ],
    ids=["package", "version", "record", "download", "repository"],
)
def test_a_resolution_for_a_different_package_cannot_vouch_for_this_install(
    session: Session, different: dict[str, str]
) -> None:
    decision = decide_registry_trust(_resolution(**different), _install(session))

    assert decision.outcome == "refuse"
    assert decision.reason == "identity_mismatch"


def test_nothing_is_trusted_when_nothing_is_installed() -> None:
    decision = decide_registry_trust(_resolution(), None)

    assert decision.outcome == "pause"
    assert decision.reason == "not_installed"


def test_an_already_trusted_package_asks_nothing(session: Session) -> None:
    decision = decide_registry_trust(_resolution(), _install(session, trusted=True))

    assert decision.outcome == "already_trusted"


def test_a_package_a_person_refused_is_never_trusted_by_the_policy(session: Session) -> None:
    install = _install(
        session,
        review={
            "reviewed_at": "2026-09-01T00:00:00+00:00",
            "trusted_by_local_user": False,
            "trust_authority": "local_user",
        },
    )

    decision = decide_registry_trust(_resolution(), install)

    assert decision.outcome == "pause"
    assert decision.reason == "previously_refused"


def test_a_refusal_recorded_before_the_authority_field_existed_is_still_a_persons(
    session: Session,
) -> None:
    """Every earlier refusal lacks `trust_authority`; ignoring them would re-trust them."""

    install = _install(
        session,
        review={
            "reviewed_at": "2026-09-01T00:00:00+00:00",
            "trusted_by_local_user": False,
        },
    )

    decision = decide_registry_trust(_resolution(), install)

    assert decision.outcome == "pause"
    assert decision.reason == "previously_refused"


@pytest.mark.parametrize("install_kind", ["already_installed", None])
def test_an_installed_package_the_registry_did_not_describe_is_not_auto_trusted(
    session: Session, install_kind: str | None
) -> None:
    decision = decide_registry_trust(_resolution(install_kind=install_kind), _install(session))

    assert decision.outcome == "pause"
    assert decision.reason == "unreviewed_source"


# ---- the recorder -------------------------------------------------------------


def _stub_verification(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def verified(current: Session, **_kwargs: Any) -> ComfyRegistryLaunchContract:
        calls.append("verified")
        return ComfyRegistryLaunchContract(("lm-atelier-registry_example",), (), ("ExampleNode",))

    monkeypatch.setattr(activation_module, "trusted_comfy_registry_launch_contract", verified)


def _record(session: Session, install_id: str, tmp_path: Path, **overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "install_id": install_id,
        "resolution": _resolution(),
        "expected_archive_sha256": _ARCHIVE,
        "expected_manifest_sha256": _MANIFEST,
        "custom_node_root": tmp_path,
        "environment_root": tmp_path,
        "media_worker_stopped": True,
    }
    arguments.update(overrides)
    return record_registry_policy_trust(session, **arguments)


def test_the_policy_grant_is_verified_and_names_the_policy_not_a_person(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _stub_verification(monkeypatch, calls)
    install = _install(session)

    state = _record(
        session, install.id, tmp_path, resolution=_resolution(warnings=("deprecated_version",))
    )

    session.refresh(install)
    assert calls == ["verified"]
    assert state.trusted is True
    assert state.active is False
    assert install.review_json["trust_authority"] == POLICY_ID
    assert install.review_json["trusted_by_local_user"] is False
    assert install.review_json["policy_notices"] == ["deprecated_version"]
    assert isinstance(install.review_json["reviewed_at"], str)


def test_the_recorder_refuses_while_the_media_worker_is_running(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _stub_verification(monkeypatch, calls)
    install = _install(session)

    with pytest.raises(ComfyRegistryActivationError) as raised:
        _record(session, install.id, tmp_path, media_worker_stopped=False)

    session.refresh(install)
    assert raised.value.code == "media_worker_running"
    assert calls == []
    assert install.trusted is False


@pytest.mark.parametrize("digest", ["expected_archive_sha256", "expected_manifest_sha256"])
def test_bytes_other_than_the_verified_ones_are_never_trusted(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, digest: str
) -> None:
    calls: list[str] = []
    _stub_verification(monkeypatch, calls)
    install = _install(session)

    with pytest.raises(ComfyRegistryActivationError) as raised:
        _record(session, install.id, tmp_path, **{digest: "f" * 64})

    session.refresh(install)
    assert raised.value.code == "registry_policy_identity_mismatch"
    assert calls == []
    assert install.trusted is False


def test_a_git_source_is_never_recorded_as_policy_trust(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _stub_verification(monkeypatch, calls)
    install = _install(session)

    with pytest.raises(ComfyRegistryActivationError) as raised:
        _record(session, install.id, tmp_path, resolution=_resolution(install_kind="git_commit"))

    session.refresh(install)
    assert raised.value.code == "registry_policy_source_review_required"
    assert calls == []
    assert install.trusted is False


def test_failed_verification_leaves_the_package_untrusted(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_current: Session, **_kwargs: Any) -> ComfyRegistryLaunchContract:
        raise RuntimeError("neutral verification failure")

    monkeypatch.setattr(activation_module, "trusted_comfy_registry_launch_contract", fail)
    install = _install(session)
    install_id = install.id

    with pytest.raises(ComfyRegistryActivationError):
        _record(session, install_id, tmp_path)

    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None
    assert stored.trusted is False
    assert "trust_authority" not in stored.review_json


def test_a_persons_existing_grant_is_left_exactly_as_it_is(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _stub_verification(monkeypatch, calls)
    review = {
        "reviewed_at": "2026-09-01T00:00:00+00:00",
        "trusted_by_local_user": True,
        "trust_authority": "local_user",
    }
    install = _install(session, trusted=True, review=dict(review))

    _record(session, install.id, tmp_path)

    session.refresh(install)
    assert install.review_json == review
    assert calls == []


def test_revoking_a_policy_grant_is_recorded_as_the_persons_and_stays_revoked(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The edge where a merge would have kept describing the policy as the authority."""

    calls: list[str] = []
    _stub_verification(monkeypatch, calls)
    install = _install(session)
    _record(session, install.id, tmp_path)

    review_comfy_registry_install(
        session,
        install_id=install.id,
        trusted=False,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
    )
    session.refresh(install)

    assert install.trusted is False
    assert install.review_json["trust_authority"] == "local_user"
    assert install.review_json["trusted_by_local_user"] is False
    assert "policy_notices" not in install.review_json
    # And the policy cannot quietly undo the person's decision.
    with pytest.raises(ComfyRegistryActivationError) as raised:
        _record(session, install.id, tmp_path)
    assert raised.value.code == "registry_policy_previously_refused"


def test_a_person_re_reviewing_a_policy_grant_becomes_the_authority(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _stub_verification(monkeypatch, calls)
    install = _install(session)
    _record(session, install.id, tmp_path, resolution=_resolution(warnings=("deprecated_version",)))

    review_comfy_registry_install(
        session,
        install_id=install.id,
        trusted=True,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
    )
    session.refresh(install)

    assert install.trusted is True
    assert install.review_json["trust_authority"] == "local_user"
    assert install.review_json["trusted_by_local_user"] is True
    assert "policy_notices" not in install.review_json
