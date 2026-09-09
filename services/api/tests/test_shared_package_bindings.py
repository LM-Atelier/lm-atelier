from __future__ import annotations

import importlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import TYPE_CHECKING

import pytest
from alembic import command
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from local_lm import db, models
from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.shared_asset_package_v1 import publish_package
from local_lm.shared_asset_store_v1 import publish_file

if TYPE_CHECKING:
    from local_lm.shared_package_bindings import SharedPackageReference


def _reference(tmp_path: Path) -> SharedPackageReference:
    api = importlib.import_module("local_lm.shared_package_bindings")
    source = tmp_path / "weights.bin"
    source.write_bytes(b"neutral immutable weights")
    root = tmp_path / "library"
    digest = publish_file(root=root, source=source)
    package_digest = publish_package(root=root, members={"unet": digest})
    return api.SharedPackageReference(
        library_id=str(uuid.uuid4()),
        consumer_id="a" * 64,
        package_digest=package_digest,
        members={"unet": digest},
    )


def test_binding_migration_preserves_existing_model_and_lora_installs(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "profile", dev=True)
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "f5c2a8d91e40")
    db.configure_database(settings)
    with db.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO model_installs"
                " (id,name,role,engine,local_path,size_bytes,compatibility,manifest_json,"
                "active,created_at,updated_at)"
                " VALUES ('model_old','Existing model','chat','mock','models/existing.gguf',"
                "7,'advanced','{}',1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO model_asset_installs"
                " (id,name,kind,local_path,size_bytes,manifest_json,active,use_case,"
                "auto_apply,default_model_strength,default_clip_strength,created_at,updated_at)"
                " VALUES ('asset_old','Existing adapter','lora','models/existing.safetensors',"
                "9,'{}',1,'',0,1,1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
            )
        )
    command.upgrade(config, "head")

    assert "shared_package_bindings" in inspect(db.engine).get_table_names(), (
        "profile migrations must provide durable shared-package bindings"
    )
    with db.engine.connect() as connection:
        assert connection.execute(
            text("SELECT local_path,active,shared_package_binding_id FROM model_installs")
        ).all() == [("models/existing.gguf", 1, None)]
        assert connection.execute(
            text("SELECT local_path,active,shared_package_binding_id FROM model_asset_installs")
        ).all() == [("models/existing.safetensors", 1, None)]
        assert (
            connection.execute(text("SELECT COUNT(*) FROM shared_package_bindings")).scalar() == 0
        )


def test_preparation_rolls_back_and_never_switches_an_existing_install(
    settings: Settings, tmp_path: Path
) -> None:
    api = importlib.import_module("local_lm.shared_package_bindings")
    reference = _reference(tmp_path)
    with db.SessionLocal() as session:
        install = models.ModelInstall(
            name="Existing model", engine="mock", local_path="models/legacy.gguf", active=True
        )
        session.add(install)
        session.commit()
        install_id = install.id
        first = api.prepare_binding(session, reference)
        assert first.state == "preparing" and first.claim_id is None
        assert api.prepare_binding(session, reference).id == first.id
        assert install.shared_package_binding_id is None
        session.rollback()

    with db.SessionLocal() as session:
        assert session.scalars(select(models.SharedPackageBinding)).all() == []
        install = session.get(models.ModelInstall, install_id)
        assert install is not None and install.active
        assert install.local_path == "models/legacy.gguf"
        binding = api.prepare_binding(session, reference)
        session.commit()
        binding_id = binding.id

    with db.SessionLocal() as session:
        binding = session.get(models.SharedPackageBinding, binding_id)
        assert binding is not None
        assert api.binding_reference(binding) == reference
        assert set(binding.member_digests_json) == {"unet"}
        assert "path" not in models.SharedPackageBinding.__table__.columns
        install = session.get(models.ModelInstall, install_id)
        assert install is not None and install.shared_package_binding_id is None


def test_both_install_kinds_keep_a_referenced_binding_durable(
    settings: Settings, tmp_path: Path
) -> None:
    api = importlib.import_module("local_lm.shared_package_bindings")
    with db.SessionLocal() as session:
        binding = api.prepare_binding(session, _reference(tmp_path))
        model = models.ModelInstall(
            name="Model",
            engine="mock",
            local_path="models/model.gguf",
            shared_package_binding_id=binding.id,
        )
        asset = models.ModelAssetInstall(
            name="Adapter",
            kind="lora",
            local_path="models/adapter.safetensors",
            shared_package_binding_id=binding.id,
        )
        session.add_all([model, asset])
        session.commit()
        binding_id = binding.id
        with pytest.raises(IntegrityError):
            session.delete(binding)
            session.commit()
        session.rollback()
        session.delete(model)
        session.commit()
        binding = session.get(models.SharedPackageBinding, binding_id)
        assert binding is not None
        with pytest.raises(IntegrityError):
            session.delete(binding)
            session.commit()
        session.rollback()
        session.delete(asset)
        session.commit()
        binding = session.get(models.SharedPackageBinding, binding_id)
        assert binding is not None
        session.delete(binding)
        session.commit()


def test_preparation_is_scoped_to_the_opaque_consumer(settings: Settings, tmp_path: Path) -> None:
    api = importlib.import_module("local_lm.shared_package_bindings")
    first = _reference(tmp_path)
    second = api.SharedPackageReference(
        library_id=first.library_id,
        consumer_id="b" * 64,
        package_digest=first.package_digest,
        members=dict(first.members),
    )
    with db.SessionLocal() as session:
        alpha = api.prepare_binding(session, first)
        beta = api.prepare_binding(session, second)
        assert alpha.id != beta.id
        assert api.prepare_binding(session, first).id == alpha.id
        session.commit()


def test_reference_keeps_a_detached_exact_member_snapshot(tmp_path: Path) -> None:
    api = importlib.import_module("local_lm.shared_package_bindings")
    original = _reference(tmp_path)
    members = dict(original.members)
    reference = api.SharedPackageReference(
        library_id=original.library_id,
        consumer_id=original.consumer_id,
        package_digest=original.package_digest,
        members=members,
    )
    members["unet"] = "f" * 64
    assert reference.members == original.members
    with pytest.raises(TypeError):
        reference.members["unet"] = "f" * 64


@pytest.mark.parametrize("state", ["release_pending", "repair_required", "unknown"])
def test_preparation_does_not_reactivate_a_retiring_or_invalid_binding(
    settings: Settings, tmp_path: Path, state: str
) -> None:
    api = importlib.import_module("local_lm.shared_package_bindings")
    reference = _reference(tmp_path)
    with db.SessionLocal() as session:
        binding = api.prepare_binding(session, reference)
        binding.state = state
        session.commit()
        with pytest.raises(api.SharedPackageBindingError, match=api.INVALID_BINDING):
            api.prepare_binding(session, reference)
        session.rollback()


def test_modified_members_and_claimless_ready_records_refuse(
    settings: Settings, tmp_path: Path
) -> None:
    api = importlib.import_module("local_lm.shared_package_bindings")
    with db.SessionLocal() as session:
        binding = api.prepare_binding(session, _reference(tmp_path))
        original = dict(binding.member_digests_json)
        binding.member_digests_json = {"unet": "f" * 64}
        with pytest.raises(api.SharedPackageBindingError, match=api.INVALID_BINDING):
            api.binding_reference(binding)
        binding.member_digests_json = original
        binding.state = "ready"
        with pytest.raises(api.SharedPackageBindingError, match=api.INVALID_BINDING):
            api.binding_reference(binding)
        session.rollback()


def test_concurrent_preparations_converge_on_one_local_record(
    settings: Settings, tmp_path: Path
) -> None:
    api = importlib.import_module("local_lm.shared_package_bindings")
    reference = _reference(tmp_path)
    start = Barrier(2)

    def prepare() -> str:
        with db.SessionLocal() as session:
            start.wait(timeout=5)
            binding = api.prepare_binding(session, reference)
            session.commit()
            return binding.id

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(prepare)
        second = executor.submit(prepare)
        assert first.result(timeout=10) == second.result(timeout=10)

    with db.SessionLocal() as session:
        assert len(session.scalars(select(models.SharedPackageBinding)).all()) == 1


def test_downgrade_refuses_live_bindings_and_keeps_legacy_installs(
    settings: Settings, tmp_path: Path
) -> None:
    api = importlib.import_module("local_lm.shared_package_bindings")
    with db.SessionLocal() as session:
        binding = api.prepare_binding(session, _reference(tmp_path))
        install = models.ModelInstall(
            name="Existing model", engine="mock", local_path="models/legacy.gguf", active=True
        )
        session.add(install)
        session.commit()
        binding_id = binding.id

    config = alembic_config(settings)
    with pytest.raises(RuntimeError, match="bindings must be released before downgrade"):
        command.downgrade(config, "f5c2a8d91e40")

    assert "shared_package_binding_id" in {
        column["name"] for column in inspect(db.engine).get_columns("model_installs")
    }
    with db.SessionLocal() as session:
        binding = session.get(models.SharedPackageBinding, binding_id)
        assert binding is not None
        session.delete(binding)
        session.commit()

    command.downgrade(config, "f5c2a8d91e40")
    assert "shared_package_bindings" not in inspect(db.engine).get_table_names()
    with db.engine.connect() as connection:
        assert connection.execute(text("SELECT local_path,active FROM model_installs")).all() == [
            ("models/legacy.gguf", 1)
        ]
