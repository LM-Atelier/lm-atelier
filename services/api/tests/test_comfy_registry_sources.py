from __future__ import annotations

import hashlib

import pytest

from local_lm.comfy_registry import ComfyNodeResolution
from local_lm.comfy_registry_sources import (
    ComfyPackageSourceError,
    resolve_comfy_package_source,
)

_PACKAGE = "example-pack"
_REVISION = "a" * 40
_REPOSITORY = "https://github.com/example/example-pack.git"
_DOWNLOAD = f"https://codeload.github.com/example/example-pack/zip/{_REVISION}"
_RECORD = (
    "github-commit:" + hashlib.sha256(f"{_REPOSITORY}{chr(0)}{_REVISION}".encode()).hexdigest()
)


def _resolution(**overrides: object) -> ComfyNodeResolution:
    values: dict[str, object] = {
        "package_id": _PACKAGE,
        "declared_version": _REVISION,
        "node_types": ("ExampleNode",),
        "install_kind": "git_commit",
        "repository_url": _REPOSITORY,
    }
    values.update(overrides)
    return ComfyNodeResolution(**values)  # type: ignore[arg-type]


def test_commit_source_accepts_its_exact_persisted_identity() -> None:
    source = resolve_comfy_package_source(
        _resolution(registry_record_id=_RECORD, download_url=_DOWNLOAD)
    )

    assert source.source_record_id == _RECORD
    assert source.download_url == _DOWNLOAD


@pytest.mark.parametrize(
    ("registry_record_id", "download_url"),
    [
        ("github-commit:" + "b" * 64, _DOWNLOAD),
        (_RECORD, f"https://codeload.github.com/example/example-pack/zip/{'b' * 40}"),
        ("registry-record", None),
        (None, "https://cdn.comfy.org/example-pack.zip"),
    ],
)
def test_commit_source_rejects_noncanonical_persisted_identity(
    registry_record_id: str | None,
    download_url: str | None,
) -> None:
    with pytest.raises(ComfyPackageSourceError, match="conflicting source metadata"):
        resolve_comfy_package_source(
            _resolution(
                registry_record_id=registry_record_id,
                download_url=download_url,
            )
        )
