from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import local_lm.shared_asset_package_v1 as package
from local_lm.filesystem_links import AnchoredDirectory, open_child_directory, open_entry
from local_lm.shared_asset_package_v1 import (
    INVALID_PACKAGE,
    MAX_PACKAGE_BYTES,
    PACKAGE_ROLES,
    PACKAGE_ROLES_V2,
    SCHEMA_ID,
    SCHEMA_ID_V2,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    SharedAssetPackageError,
    load_package,
    publish_package,
)
from local_lm.shared_asset_root_v1 import default_shared_asset_root
from local_lm.shared_asset_store_v1 import object_path, publish_file


def _published(tmp_path: Path, name: str, payload: bytes) -> tuple[Path, str]:
    root = tmp_path / "packages"
    source = tmp_path / name
    source.write_bytes(payload)
    digest = publish_file(root=root, source=source)
    return root, digest


def test_package_maps_closed_roles_to_published_digests(tmp_path: Path) -> None:
    root, weights = _published(tmp_path, "model.safetensors", b"weights")
    source = tmp_path / "lora.bin"
    source.write_bytes(b"adapter")
    lora = publish_file(root=root, source=source)

    digest = publish_package(root=root, members={"unet": weights, "lora": lora})

    assert load_package(root=root, digest=digest) == (("lora", lora), ("unet", weights))
    assert object_path(root=root, digest=digest).is_file()


def test_package_schema_identity_and_version_are_pinned(tmp_path: Path) -> None:
    root, weights = _published(tmp_path, "model.bin", b"weights")
    digest = publish_package(root=root, members={"unet": weights})
    value = json.loads(object_path(root=root, digest=digest).read_text(encoding="ascii"))

    assert SCHEMA_ID == "lm-atelier-shared-asset-package-v1"
    assert SCHEMA_VERSION == 1
    assert value["schema"] == SCHEMA_ID
    assert value["version"] == SCHEMA_VERSION


@pytest.mark.parametrize(
    ("schema", "version"),
    [
        ("lm-atelier-shared-asset-package-v2", 1),
        (SCHEMA_ID, 2),
        (SCHEMA_ID, "1"),
        (SCHEMA_ID, True),
        (SCHEMA_ID_V2, True),
    ],
)
def test_load_refuses_other_schema_identities_and_versions(
    tmp_path: Path, schema: object, version: object
) -> None:
    root, weights = _published(tmp_path, "model.bin", b"weights")
    descriptor = tmp_path / "package.json"
    descriptor.write_text(
        json.dumps(
            {"members": {"unet": weights}, "schema": schema, "version": version},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="ascii",
    )
    digest = publish_file(root=root, source=descriptor)

    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        load_package(root=root, digest=digest)


def test_load_refuses_noncanonical_v1_and_v2_documents(tmp_path: Path) -> None:
    root, weights = _published(tmp_path, "model.bin", b"weights")
    workflow_source = tmp_path / "workflow.json"
    workflow_source.write_bytes(b"workflow")
    workflow = publish_file(root=root, source=workflow_source)
    documents = (
        {
            "schema": SCHEMA_ID,
            "version": SCHEMA_VERSION,
            "members": {"unet": weights},
        },
        {
            "schema": SCHEMA_ID_V2,
            "version": SCHEMA_VERSION_V2,
            "members": {"workflow": workflow, "unet": weights},
        },
    )

    for index, value in enumerate(documents):
        descriptor = tmp_path / f"noncanonical-{index}.json"
        descriptor.write_text(json.dumps(value), encoding="ascii")
        digest = publish_file(root=root, source=descriptor)

        with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
            load_package(root=root, digest=digest)


@pytest.mark.parametrize("failure", [RecursionError, OverflowError])
def test_package_json_failures_expose_only_the_fixed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    root, _weights = _published(tmp_path, "ok.bin", b"ok")
    descriptor = tmp_path / "package.json"
    descriptor.write_bytes(b"{}")
    digest = publish_file(root=root, source=descriptor)

    def fail_decode(_raw: object) -> object:
        raise failure

    monkeypatch.setattr(json, "loads", fail_decode)

    with pytest.raises(SharedAssetPackageError) as caught:
        load_package(root=root, digest=digest)
    assert type(caught.value) is SharedAssetPackageError
    assert str(caught.value) == INVALID_PACKAGE
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_package_does_not_discover_or_write_the_desktop_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    root, weights = _published(tmp_path, "only.bin", b"keep-out")

    digest = publish_package(root=root, members={"vae": weights})

    desktop = default_shared_asset_root()
    assert object_path(root=root, digest=digest).is_file()
    assert not (desktop / digest[:2] / digest[2:4] / digest).exists()


@pytest.mark.parametrize("role", ["../evil", "unet/path", "future_role", ""])
def test_package_refuses_roles_outside_the_closed_vocabulary(tmp_path: Path, role: str) -> None:
    root, weights = _published(tmp_path, "ok.bin", b"ok")

    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        publish_package(root=root, members={role: weights})


def test_package_refuses_invalid_missing_and_drifted_members(tmp_path: Path) -> None:
    root, weights = _published(tmp_path, "ok.bin", b"ok-bytes")
    stored = object_path(root=root, digest=weights)

    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        publish_package(root=root, members={"unet": "ab" * 32})
    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        publish_package(root=root, members={})

    stored.write_bytes(b"corrupted")
    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        publish_package(root=root, members={"unet": weights})


def test_load_revalidates_package_and_every_member(tmp_path: Path) -> None:
    root, weights = _published(tmp_path, "ok.bin", b"ok-bytes")
    digest = publish_package(root=root, members={"unet": weights})

    object_path(root=root, digest=weights).write_bytes(b"corrupted")
    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        load_package(root=root, digest=digest)

    object_path(root=root, digest=weights).write_bytes(b"ok-bytes")
    package_path = object_path(root=root, digest=digest)
    package_path.write_bytes(b"corrupted package")
    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        load_package(root=root, digest=digest)


def test_package_document_is_bounded_before_json_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _weights = _published(tmp_path, "ok.bin", b"ok")
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b" " * MAX_PACKAGE_BYTES + b"}")
    digest = publish_file(root=root, source=oversized)

    def decode_was_reached(_raw: object) -> object:
        raise AssertionError("oversized package reached JSON decoding")

    monkeypatch.setattr(json, "loads", decode_was_reached)

    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        load_package(root=root, digest=digest)


def test_member_read_uses_the_held_directory_not_its_path_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed path label cannot redirect the descriptor-backed read."""

    original = b"member bytes"
    root, weights = _published(tmp_path, "ok.bin", original)
    digest = publish_package(root=root, members={"unet": weights})
    object_path(root=root, digest=weights).write_bytes(b"drifted")

    outside = tmp_path / "outside" / weights[:2] / weights[2:4]
    outside.mkdir(parents=True)
    (outside / weights).write_bytes(original)
    real_open_child = open_child_directory

    def relabel(anchor: AnchoredDirectory, name: str, *, create: bool = False) -> AnchoredDirectory:
        child = real_open_child(anchor, name, create=create)
        if anchor.path.name == weights[:2] and name == weights[2:4]:
            child.path = outside
        return child

    monkeypatch.setattr(package, "open_child_directory", relabel)

    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        load_package(root=root, digest=digest)


def test_package_publication_is_create_only_and_convergent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, weights = _published(tmp_path, "ok.bin", b"ok")
    first = publish_package(root=root, members={"unet": weights})
    real_open = open_entry
    opened = 0

    def count_open(anchor: AnchoredDirectory, name: str) -> int | None:
        nonlocal opened
        descriptor = real_open(anchor, name)
        if name == first and descriptor is not None:
            opened += 1
        return descriptor

    monkeypatch.setattr(package, "open_entry", count_open)

    second = publish_package(root=root, members={"unet": weights})

    assert second == first
    assert opened >= 1
    assert object_path(root=root, digest=first).read_bytes()


def test_symlinked_member_entry_is_refused_without_following_it(tmp_path: Path) -> None:
    root, weights = _published(tmp_path, "ok.bin", b"ok-bytes")
    digest = publish_package(root=root, members={"unet": weights})
    member = object_path(root=root, digest=weights)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"ok-bytes")
    member.unlink()
    try:
        os.symlink(outside, member)
    except OSError:
        pytest.skip("this host does not allow file links")

    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        load_package(root=root, digest=digest)


def test_invalid_inputs_expose_only_the_fixed_package_error(tmp_path: Path) -> None:
    root, _weights = _published(tmp_path, "ok.bin", b"ok")
    cases = (
        lambda: load_package(root=root, digest="../../evil"),
        lambda: load_package(root=root, digest="f" * 64),
        lambda: publish_package(root=Path("relative"), members={"unet": "f" * 64}),
    )

    for invoke in cases:
        with pytest.raises(SharedAssetPackageError) as caught:
            invoke()
        assert type(caught.value) is SharedAssetPackageError
        assert str(caught.value) == INVALID_PACKAGE
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_package_digest_is_the_canonical_document_hash(tmp_path: Path) -> None:
    root, weights = _published(tmp_path, "ok.bin", b"ok")
    digest = publish_package(root=root, members={"unet": weights})
    payload = object_path(root=root, digest=digest).read_bytes()

    assert hashlib.sha256(payload).hexdigest() == digest


def test_package_v1_wire_vocabulary_and_bytes_remain_frozen(tmp_path: Path) -> None:
    assert (
        frozenset(
            {
                "checkpoint",
                "clip_vision",
                "controlnet",
                "diffusion_model",
                "embedding",
                "gguf_model",
                "ip_adapter",
                "lora",
                "text_encoder",
                "unet",
                "upscaler",
                "vae",
            }
        )
        == PACKAGE_ROLES
    )
    assert "workflow" not in PACKAGE_ROLES
    assert PACKAGE_ROLES | {"workflow"} == PACKAGE_ROLES_V2

    root, weights = _published(tmp_path, "model.bin", b"weights")
    digest = publish_package(root=root, members={"unet": weights})
    payload = object_path(root=root, digest=digest).read_bytes()
    expected = json.dumps(
        {
            "members": {"unet": weights},
            "schema": SCHEMA_ID,
            "version": SCHEMA_VERSION,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")

    assert payload == expected
    assert hashlib.sha256(expected).hexdigest() == digest


def test_package_v2_is_selected_only_for_workflow_members(tmp_path: Path) -> None:
    assert SCHEMA_ID_V2 == "lm-atelier-shared-asset-package-v2"
    assert SCHEMA_VERSION_V2 == 2
    root, workflow = _published(tmp_path, "workflow.json", b"workflow")
    model_source = tmp_path / "model.bin"
    model_source.write_bytes(b"weights")
    model = publish_file(root=root, source=model_source)
    digest = publish_package(root=root, members={"workflow": workflow, "unet": model})
    payload = object_path(root=root, digest=digest).read_bytes()

    expected = json.dumps(
        {
            "members": {"unet": model, "workflow": workflow},
            "schema": "lm-atelier-shared-asset-package-v2",
            "version": 2,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert payload == expected
    assert hashlib.sha256(payload).hexdigest() == digest
    assert load_package(root=root, digest=digest) == (
        ("unet", model),
        ("workflow", workflow),
    )


def test_package_v2_refuses_a_document_without_a_workflow_member(
    tmp_path: Path,
) -> None:
    root, weights = _published(tmp_path, "model.bin", b"weights")
    descriptor = tmp_path / "package-v2.json"
    descriptor.write_text(
        json.dumps(
            {
                "members": {"unet": weights},
                "schema": SCHEMA_ID_V2,
                "version": SCHEMA_VERSION_V2,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="ascii",
    )
    digest = publish_file(root=root, source=descriptor)

    with pytest.raises(SharedAssetPackageError, match=INVALID_PACKAGE):
        load_package(root=root, digest=digest)
