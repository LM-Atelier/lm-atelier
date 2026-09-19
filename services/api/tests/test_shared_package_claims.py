from __future__ import annotations

from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Never

import pytest

from local_lm import db, models
from local_lm import shared_package_claims as api
from local_lm.config import Settings
from local_lm.filesystem_links import AnchoredDirectory
from local_lm.shared_asset_contract_v1 import initialize_store_identity
from local_lm.shared_asset_lock_v1 import SharedAssetLockError
from local_lm.shared_asset_package_v1 import _load_package_from_store_versions, publish_package
from local_lm.shared_asset_registry_v1 import (
    FINAL,
    claims_for_consumer,
    finalize_claim,
    release_claim,
    reserve_claim,
)
from local_lm.shared_asset_store_v1 import object_path, publish_file
from local_lm.shared_package_bindings import (
    SharedPackageBindingError,
    SharedPackageReference,
    prepare_binding,
)


def _prepared(tmp_path: Path, consumer: str = "a" * 64) -> tuple[Path, str, SharedPackageReference]:
    root = tmp_path / "library"
    identity = initialize_store_identity(root=root)
    source = tmp_path / "weights.bin"
    source.write_bytes(b"neutral model weights")
    digest = publish_file(root=root, source=source)
    package = publish_package(root=root, members={"unet": digest})
    reference = SharedPackageReference(identity.library_uuid, consumer, package, {"unet": digest})
    with db.SessionLocal() as session:
        binding = prepare_binding(session, reference)
        session.commit()
        binding_id = binding.id
    return root, binding_id, reference


def _row(binding_id: str) -> tuple[str, str | None] | None:
    with db.SessionLocal() as session:
        row = session.get(models.SharedPackageBinding, binding_id)
        return None if row is None else (row.state, row.claim_id)


def test_completion_is_durable_idempotent_and_keeps_old_install(
    settings: Settings, tmp_path: Path
) -> None:
    root, binding_id, reference = _prepared(tmp_path)
    with db.SessionLocal() as session:
        install = models.ModelInstall(
            name="Existing", engine="mock", local_path="models/old.gguf", active=True
        )
        session.add(install)
        session.commit()
        install_id = install.id
    claim = api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    assert (
        api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
        == claim
    )
    assert _row(binding_id) == ("ready", claim)
    claims = claims_for_consumer(database=root / "index.sqlite3", consumer_id=reference.consumer_id)
    assert [(item.claim_id, item.state) for item in claims] == [(claim, FINAL)]
    with db.SessionLocal() as session:
        persisted = session.get(models.ModelInstall, install_id)
        assert persisted is not None
        assert persisted.active and persisted.local_path == "models/old.gguf"
        assert persisted.shared_package_binding_id is None


@pytest.mark.parametrize(
    "boundary", ["reserve_claim", "_load_package_from_store_versions", "finalize_claim"]
)
def test_completion_recovers_after_external_boundary(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    root, binding_id, reference = _prepared(tmp_path)
    original = getattr(api, boundary)
    assert callable(original)

    def interrupted(**kwargs: object) -> Never:
        original(**kwargs)
        raise RuntimeError("constructed interruption")

    with monkeypatch.context() as patch:
        patch.setattr(api, boundary, interrupted)
        with pytest.raises(RuntimeError, match="constructed interruption"):
            api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    claims = claims_for_consumer(database=root / "index.sqlite3", consumer_id=reference.consumer_id)
    assert len(claims) == 1
    claim = api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    assert claim == claims[0].claim_id
    assert _row(binding_id) == ("ready", claim)


def test_member_verification_has_no_profile_or_registry_writer(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, binding_id, reference = _prepared(tmp_path)
    original = _load_package_from_store_versions
    observed = []

    def verify(
        *, store: AnchoredDirectory, digest: str, allow_v2: bool
    ) -> tuple[tuple[str, str], ...]:
        with db.SessionLocal() as session:
            session.connection().exec_driver_sql("UPDATE shared_package_bindings SET id=id WHERE 0")
            session.rollback()
        reserve_claim(
            database=root / "index.sqlite3",
            consumer_id="b" * 64,
            package_digest=reference.package_digest,
        )
        observed.append(_row(binding_id))
        return original(store=store, digest=digest, allow_v2=allow_v2)

    monkeypatch.setattr(api, "_load_package_from_store_versions", verify)
    claim = api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    assert observed == [("preparing", claim)]


@pytest.mark.parametrize("kind", ["model", "asset"])
def test_release_refuses_each_referenced_install(
    settings: Settings, tmp_path: Path, kind: str
) -> None:
    root, binding_id, reference = _prepared(tmp_path)
    claim = api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    with db.SessionLocal() as session:
        row: models.ModelInstall | models.ModelAssetInstall
        if kind == "model":
            row = models.ModelInstall(
                name="Model",
                engine="mock",
                local_path="models/old.gguf",
                shared_package_binding_id=binding_id,
            )
        else:
            row = models.ModelAssetInstall(
                name="Adapter",
                kind="lora",
                local_path="models/old.bin",
                shared_package_binding_id=binding_id,
            )
        session.add(row)
        session.commit()
    with pytest.raises(SharedPackageBindingError):
        api.release_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    assert _row(binding_id) == ("ready", claim)
    assert (
        len(claims_for_consumer(database=root / "index.sqlite3", consumer_id=reference.consumer_id))
        == 1
    )


@pytest.mark.parametrize("after_release", [False, True])
def test_release_recovers_and_preserves_other_consumer_and_bytes(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_release: bool
) -> None:
    root, binding_id, reference = _prepared(tmp_path)
    claim = api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    other = reserve_claim(
        database=root / "index.sqlite3",
        consumer_id="b" * 64,
        package_digest=reference.package_digest,
    )
    original = release_claim

    def interrupted(*, database: Path, consumer_id: str, claim_id: str) -> Never:
        if after_release:
            original(database=database, consumer_id=consumer_id, claim_id=claim_id)
        raise RuntimeError("constructed interruption")

    with monkeypatch.context() as patch:
        patch.setattr(api, "release_claim", interrupted)
        with pytest.raises(RuntimeError, match="constructed interruption"):
            api.release_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    assert _row(binding_id) == ("release_pending", claim)
    api.release_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    api.release_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    assert _row(binding_id) is None
    assert (
        claims_for_consumer(database=root / "index.sqlite3", consumer_id=reference.consumer_id)
        == []
    )
    assert claims_for_consumer(database=root / "index.sqlite3", consumer_id="b" * 64) == [other]
    assert (
        object_path(root=root, digest=reference.members["unet"]).read_bytes()
        == b"neutral model weights"
    )


def test_release_recovers_reservation_before_local_claim_commit(
    settings: Settings, tmp_path: Path
) -> None:
    root, binding_id, reference = _prepared(tmp_path)
    reserve_claim(
        database=root / "index.sqlite3",
        consumer_id=reference.consumer_id,
        package_digest=reference.package_digest,
    )
    api.release_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    assert _row(binding_id) is None
    assert (
        claims_for_consumer(database=root / "index.sqlite3", consumer_id=reference.consumer_id)
        == []
    )


def test_release_refuses_an_unexplained_finalized_claim(settings: Settings, tmp_path: Path) -> None:
    """The negative of the reservation-recovery case above.

    That one proves a PROVISIONAL reservation is adopted when the local row lost
    its claim to a crash. The refusal beside it - only a provisional claim may be
    adopted, never a finalized one - had nothing holding it: deleting it left the
    whole module green, so the rule could have been removed by an unrelated edit
    without anything noticing.

    A finalized claim the local record cannot account for is not this operation's
    to release. Adopting one leads straight to releasing it, which is the single
    thing this module promises not to do to bytes another consumer may hold.
    """

    root, binding_id, reference = _prepared(tmp_path)
    claim = reserve_claim(
        database=root / "index.sqlite3",
        consumer_id=reference.consumer_id,
        package_digest=reference.package_digest,
    )
    finalize_claim(
        database=root / "index.sqlite3",
        consumer_id=reference.consumer_id,
        claim_id=claim.claim_id,
    )

    with pytest.raises(SharedPackageBindingError):
        api.release_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)

    # Surviving is not enough: a refusal that had already moved the row, or
    # already written a claim onto it, would leave the next release believing
    # the finalized claim was its own. The binding has to be exactly as it was.
    assert _row(binding_id) == ("preparing", None), "the refusal must leave the binding untouched"
    remaining = claims_for_consumer(
        database=root / "index.sqlite3", consumer_id=reference.consumer_id
    )
    assert [item.claim_id for item in remaining] == [claim.claim_id]
    assert remaining[0].state == FINAL, "the finalized claim must not have been released"


def test_wrong_library_refuses_before_claim_mutation(settings: Settings, tmp_path: Path) -> None:
    root, binding_id, reference = _prepared(tmp_path)
    wrong = tmp_path / "other-library"
    initialize_store_identity(root=wrong)
    with pytest.raises(SharedPackageBindingError):
        api.complete_binding_claim(sessions=db.SessionLocal, root=wrong, binding_id=binding_id)
    assert _row(binding_id) == ("preparing", None)
    assert not (wrong / "index.sqlite3").exists()


def test_corrupt_member_keeps_recovery_claim(settings: Settings, tmp_path: Path) -> None:
    root, binding_id, reference = _prepared(tmp_path)
    object_path(root=root, digest=reference.members["unet"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    row = _row(binding_id)
    assert row is not None
    state, claim = row
    assert state == "preparing" and claim
    assert (
        len(claims_for_consumer(database=root / "index.sqlite3", consumer_id=reference.consumer_id))
        == 1
    )


def test_release_cannot_enter_while_completion_holds_binding_lock(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, binding_id, reference = _prepared(tmp_path)
    original = _load_package_from_store_versions

    def verify(
        *, store: AnchoredDirectory, digest: str, allow_v2: bool
    ) -> tuple[tuple[str, str], ...]:
        with pytest.raises(SharedAssetLockError):
            api.release_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
        return original(store=store, digest=digest, allow_v2=allow_v2)

    monkeypatch.setattr(api, "_load_package_from_store_versions", verify)
    claim = api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    assert _row(binding_id) == ("ready", claim)


@pytest.mark.parametrize("operation", ["complete", "release"])
@pytest.mark.parametrize("commit_number", [1, 2])
@pytest.mark.parametrize("after_commit", [False, True])
def test_recovery_at_each_profile_commit(
    settings: Settings, tmp_path: Path, operation: str, commit_number: int, after_commit: bool
) -> None:
    from sqlalchemy.orm import Session, sessionmaker

    root, binding_id, reference = _prepared(tmp_path)
    if operation == "release":
        api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    calls = 0

    class InterruptedSession(Session):
        def commit(self) -> None:
            nonlocal calls
            calls += 1
            if calls == commit_number and not after_commit:
                raise RuntimeError("constructed commit interruption")
            super().commit()
            if calls == commit_number and after_commit:
                raise RuntimeError("constructed commit interruption")

    factory = sessionmaker(bind=db.engine, class_=InterruptedSession)
    action = api.complete_binding_claim if operation == "complete" else api.release_binding_claim
    with pytest.raises(RuntimeError, match="constructed commit interruption"):
        action(sessions=factory, root=root, binding_id=binding_id)
    action(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    claims = claims_for_consumer(database=root / "index.sqlite3", consumer_id=reference.consumer_id)
    if operation == "complete":
        assert len(claims) == 1 and claims[0].state == FINAL
        assert _row(binding_id) == ("ready", claims[0].claim_id)
    else:
        assert _row(binding_id) is None and claims == []


def test_install_added_after_release_intent_preserves_claim(
    settings: Settings, tmp_path: Path
) -> None:
    from sqlalchemy.orm import Session, sessionmaker

    root, binding_id, reference = _prepared(tmp_path)
    claim = api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    inserted = False

    class ReferencingSession(Session):
        def commit(self) -> None:
            nonlocal inserted
            super().commit()
            if not inserted:
                inserted = True
                with db.SessionLocal() as other:
                    other.add(
                        models.ModelInstall(
                            name="Concurrent",
                            engine="mock",
                            local_path="models/existing.gguf",
                            shared_package_binding_id=binding_id,
                        )
                    )
                    other.commit()

    with pytest.raises(SharedPackageBindingError):
        api.release_binding_claim(
            sessions=sessionmaker(bind=db.engine, class_=ReferencingSession),
            root=root,
            binding_id=binding_id,
        )
    assert _row(binding_id) == ("release_pending", claim)
    assert [
        item.claim_id
        for item in claims_for_consumer(
            database=root / "index.sqlite3",
            consumer_id=reference.consumer_id,
        )
    ] == [claim]


def test_profile_writer_spans_external_release(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    root, binding_id, reference = _prepared(tmp_path)
    api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    original = release_claim
    observed = []

    def release(*, database: Path, consumer_id: str, claim_id: str) -> None:
        connection = sqlite3.connect(str(db.engine.url.database), timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                connection.execute("BEGIN IMMEDIATE")
            observed.append(True)
        finally:
            connection.close()
        return original(database=database, consumer_id=consumer_id, claim_id=claim_id)

    monkeypatch.setattr(api, "release_claim", release)
    api.release_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
    assert observed == [True] and _row(binding_id) is None


def _complete_in_child(
    profile: str, root: str, binding_id: str, entered: Event, resume: Event, result: Queue[str]
) -> None:
    from local_lm import shared_package_claims as api

    db.configure_database(Settings(data_dir=Path(profile), dev=True))
    original = _load_package_from_store_versions

    def verify(
        *, store: AnchoredDirectory, digest: str, allow_v2: bool
    ) -> tuple[tuple[str, str], ...]:
        entered.set()
        if not resume.wait(20):
            raise RuntimeError("constructed child timeout")
        return original(store=store, digest=digest, allow_v2=allow_v2)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api, "_load_package_from_store_versions", verify)
        try:
            result.put(
                api.complete_binding_claim(
                    sessions=db.SessionLocal,
                    root=Path(root),
                    binding_id=binding_id,
                )
            )
        finally:
            db.engine.dispose()


def test_binding_operations_serialize_across_processes(settings: Settings, tmp_path: Path) -> None:
    import multiprocessing

    from local_lm.shared_asset_lock_v1 import SharedAssetLockError

    root, binding_id, reference = _prepared(tmp_path)
    context = multiprocessing.get_context("spawn")
    entered, resume, result = context.Event(), context.Event(), context.Queue()
    child = context.Process(
        target=_complete_in_child,
        args=(str(settings.data_dir), str(root), binding_id, entered, resume, result),
    )
    child.start()
    try:
        assert entered.wait(20)
        with pytest.raises(SharedAssetLockError):
            api.release_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
        with pytest.raises(SharedAssetLockError):
            api.complete_binding_claim(sessions=db.SessionLocal, root=root, binding_id=binding_id)
        resume.set()
        claim = result.get(timeout=20)
        child.join(20)
        assert child.exitcode == 0
        assert _row(binding_id) == ("ready", claim)
    finally:
        resume.set()
        child.join(5)
        if child.is_alive():
            child.terminate()
            child.join(5)
        result.close()
        result.join_thread()
