"""Bind remote downloads and retained reviewed wheels without conflating their origins."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from email import policy
from email.parser import BytesParser
from typing import Any, NoReturn

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import parse_tag
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .comfy_registry_source_artifacts import (
    MAX_REVIEWED_SOURCE_WHEEL_BYTES,
    MAX_SOURCE_DECLARATION_CHARACTERS,
    VerifiedSourceWheel,
    verified_reviewed_source_wheel,
)
from .comfy_registry_wheel_artifacts import (
    MAX_WHEEL_MANIFEST_ARTIFACTS,
    ComfyRegistryWheelArtifact,
    ComfyRegistryWheelArtifactError,
    ComfyRegistryWheelArtifactManifest,
    build_comfy_registry_wheel_artifact_manifest,
    comfy_registry_wheel_target_sha256,
    validate_comfy_registry_wheel_artifact_manifest,
)
from .package_sources import classify_source_url

MAX_REVIEWED_INPUT_TAGS = 4_096


class ComfyRegistryWheelInputError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__("Wheel input identity is invalid or incompatible with the target.")
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryReviewedWheelInput:
    declaration: str
    declaration_sha256: str
    repository: str
    commit: str
    artifact_id: str
    sha256: str
    size_bytes: int
    review_sha256: str
    name: str
    version: str
    filename: str
    metadata_sha256: str
    wheel_tags: tuple[str, ...]
    compatibility_tag: str
    target_sha256: str


@dataclass(frozen=True)
class ComfyRegistryWheelInputManifest:
    version: int
    declaration_sha256: str
    remote: ComfyRegistryWheelArtifactManifest
    reviewed_local: tuple[ComfyRegistryReviewedWheelInput, ...]
    manifest_sha256: str


def _fail(code: str = "invalid_wheel_input") -> NoReturn:
    raise ComfyRegistryWheelInputError(code)


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail()
    return value


def _bounded_filename_tags(filename: str) -> bool:
    groups = filename.removesuffix(".whl").rsplit("-", 3)
    count = 1
    for group in groups[-3:]:
        count *= len(group.split("."))
    return len(groups) == 4 and count <= MAX_REVIEWED_INPUT_TAGS


def _validate_local(item: ComfyRegistryReviewedWheelInput) -> None:
    if not isinstance(item, ComfyRegistryReviewedWheelInput):
        _fail()
    for value in (
        item.declaration_sha256,
        item.sha256,
        item.review_sha256,
        item.metadata_sha256,
        item.target_sha256,
    ):
        _digest(value)
    if (
        not isinstance(item.declaration, str)
        or not 1 <= len(item.declaration) <= MAX_SOURCE_DECLARATION_CHARACTERS
        or any(ord(character) < 32 or ord(character) == 127 for character in item.declaration)
        or not isinstance(item.artifact_id, str)
        or item.artifact_id != f"sha256:{item.sha256}"
        or type(item.size_bytes) is not int
        or not 0 < item.size_bytes <= MAX_REVIEWED_SOURCE_WHEEL_BYTES
        or not isinstance(item.filename, str)
        or not 1 <= len(item.filename) <= 500
        or "/" in item.filename
        or "\\" in item.filename
        or not _bounded_filename_tags(item.filename)
        or not isinstance(item.wheel_tags, tuple)
        or len(item.wheel_tags) > MAX_REVIEWED_INPUT_TAGS
    ):
        _fail()
    try:
        requirement = Requirement(item.declaration)
        if requirement.url is None:
            _fail()
        source = classify_source_url(requirement.url)
        name, version, _build, tags = parse_wheel_filename(item.filename)
    except (InvalidRequirement, InvalidWheelFilename, InvalidVersion, ValueError):
        _fail()
    if (
        str(requirement) != item.declaration
        or hashlib.sha256(item.declaration.encode("utf-8")).hexdigest() != item.declaration_sha256
        or source.repository is None
        or source.commit is None
        or source.reference is not None
        or (item.repository, item.commit) != (source.repository, source.commit)
        or item.name != canonicalize_name(requirement.name)
        or item.name != str(name)
        or item.version != str(version)
        or item.wheel_tags != tuple(sorted(str(tag) for tag in tags))
        or item.compatibility_tag not in item.wheel_tags
    ):
        _fail()


def reviewed_wheel_input(
    session: Session,
    store: ArtifactStore,
    *,
    declaration: str,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
) -> ComfyRegistryReviewedWheelInput:
    """Bind verified bytes to a compatible target; staging must revalidate their review."""
    wheel = verified_reviewed_source_wheel(session, store, declaration=declaration)
    return wheel_input_from_verified_source(
        wheel, marker_environment=marker_environment, supported_tags=supported_tags
    )


def wheel_input_from_verified_source(
    wheel: VerifiedSourceWheel,
    *,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
) -> ComfyRegistryReviewedWheelInput:
    """Describe freshly verified source bytes without granting a new review."""
    if not _bounded_filename_tags(wheel.filename):
        _fail()
    try:
        target = comfy_registry_wheel_target_sha256(marker_environment, supported_tags)
        _name, _version, _build, tags = parse_wheel_filename(wheel.filename)
    except (ComfyRegistryWheelArtifactError, InvalidWheelFilename, InvalidVersion):
        _fail()
    wheel_tags = tuple(sorted(str(tag) for tag in tags))
    target_tags = (str(tag) for value in supported_tags for tag in parse_tag(value))
    compatible = next((tag for tag in target_tags if tag in wheel_tags), None)
    if compatible is None:
        _fail("source_wheel_target_incompatible")
    metadata_sha256 = hashlib.sha256(wheel.core_metadata).hexdigest()
    headers = BytesParser(policy=policy.default).parsebytes(wheel.core_metadata, headersonly=True)
    python_requirements = headers.get_all("Requires-Python", [])
    if len(python_requirements) > 1:
        _fail("source_wheel_python_incompatible")
    if python_requirements:
        try:
            allowed = SpecifierSet(str(python_requirements[0]))
            python = Version(marker_environment["python_full_version"])
        except (InvalidSpecifier, InvalidVersion, KeyError):
            _fail("source_wheel_python_incompatible")
        if not allowed.contains(python, prereleases=True):
            _fail("source_wheel_python_incompatible")
    item = ComfyRegistryReviewedWheelInput(
        wheel.declaration,
        hashlib.sha256(wheel.declaration.encode("utf-8")).hexdigest(),
        wheel.repository,
        wheel.commit,
        wheel.artifact_id,
        wheel.artifact_sha256,
        len(wheel.payload),
        wheel.review_sha256,
        wheel.distribution,
        wheel.version,
        wheel.filename,
        metadata_sha256,
        wheel_tags,
        compatible,
        target,
    )
    _validate_local(item)
    return item


def build_comfy_registry_wheel_input_manifest(
    declaration_sha256: str,
    remote: ComfyRegistryWheelArtifactManifest,
    reviewed_local: Sequence[ComfyRegistryReviewedWheelInput],
) -> ComfyRegistryWheelInputManifest:
    """Bind a mixed input set while retaining the unchanged remote manifest contract."""
    _digest(declaration_sha256)
    try:
        artifacts = validate_comfy_registry_wheel_artifact_manifest(remote)
    except ComfyRegistryWheelArtifactError:
        _fail("invalid_remote_wheel_inputs")
    if isinstance(reviewed_local, str | bytes) or not isinstance(reviewed_local, Sequence):
        _fail()
    if len(artifacts) + len(reviewed_local) > MAX_WHEEL_MANIFEST_ARTIFACTS:
        _fail("too_many_wheel_inputs")
    for item in reviewed_local:
        _validate_local(item)
        if item.target_sha256 != remote.target_sha256:
            _fail("wheel_input_target_mismatch")
    local = tuple(sorted(reviewed_local, key=lambda item: item.name))
    names = [item.name for item in artifacts] + [item.name for item in local]
    filenames = [item.filename for item in artifacts] + [item.filename for item in local]
    if len(names) != len(set(names)) or len(filenames) != len(set(filenames)):
        _fail("duplicate_wheel_input")
    payload = _manifest_payload(declaration_sha256, remote, local)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return ComfyRegistryWheelInputManifest(
        1, declaration_sha256, remote, local, hashlib.sha256(encoded).hexdigest()
    )


def validate_comfy_registry_wheel_input_manifest(
    manifest: ComfyRegistryWheelInputManifest,
) -> ComfyRegistryWheelInputManifest:
    """Recompute every input identity without granting fresh artifact-review authority."""
    if (
        not isinstance(manifest, ComfyRegistryWheelInputManifest)
        or type(manifest.version) is not int
        or manifest.version != 1
        or not isinstance(manifest.reviewed_local, tuple)
    ):
        _fail()
    expected = build_comfy_registry_wheel_input_manifest(
        manifest.declaration_sha256, manifest.remote, manifest.reviewed_local
    )
    if manifest.reviewed_local != expected.reviewed_local:
        _fail()
    if not hmac.compare_digest(_digest(manifest.manifest_sha256), expected.manifest_sha256):
        _fail("wheel_input_manifest_changed")
    return expected


def _manifest_payload(
    declaration: str,
    remote: ComfyRegistryWheelArtifactManifest,
    local: tuple[ComfyRegistryReviewedWheelInput, ...],
) -> dict[str, object]:
    entries = [{"kind": "remote", **asdict(artifact)} for artifact in remote.artifacts] + [
        {"kind": "reviewed-local", **asdict(item)} for item in local
    ]
    for entry in entries:
        entry["wheel_tags"] = list(entry["wheel_tags"])
    return {
        "version": 1,
        "declaration_sha256": declaration,
        "target_sha256": remote.target_sha256,
        "remote_declaration_sha256": remote.declaration_sha256,
        "remote_manifest_sha256": remote.manifest_sha256,
        "inputs": sorted(entries, key=lambda entry: str(entry["name"])),
    }


def wheel_input_manifest_payload(manifest: ComfyRegistryWheelInputManifest) -> dict[str, object]:
    """Serialize a validated input identity without retained bytes or local paths."""
    checked = validate_comfy_registry_wheel_input_manifest(manifest)
    return {
        **_manifest_payload(checked.declaration_sha256, checked.remote, checked.reviewed_local),
        "manifest_sha256": checked.manifest_sha256,
    }


def _entry_fields(value: dict[str, Any], expected: set[str]) -> dict[str, Any]:
    if set(value) != expected | {"kind"}:
        _fail()
    for key in expected - {"wheel_tags", "size_bytes"}:
        item = value[key]
        if key == "metadata_sha256" and value["kind"] == "remote" and item is None:
            continue
        if not isinstance(item, str) or len(item) > MAX_SOURCE_DECLARATION_CHARACTERS:
            _fail()
    if type(value["size_bytes"]) is not int:
        _fail()
    if not 1 <= len(value["filename"]) <= 500 or not _bounded_filename_tags(value["filename"]):
        _fail()
    tags = value["wheel_tags"]
    if (
        not isinstance(tags, list)
        or len(tags) > MAX_REVIEWED_INPUT_TAGS
        or not all(isinstance(tag, str) for tag in tags)
    ):
        _fail()
    return {
        **{key: item for key, item in value.items() if key != "kind"},
        "wheel_tags": tuple(tags),
    }


def parse_wheel_input_manifest(value: object) -> ComfyRegistryWheelInputManifest:
    """Refuse unknown fields and changed identities when reading a persisted input manifest."""
    if not isinstance(value, dict) or set(value) != {
        "version",
        "declaration_sha256",
        "target_sha256",
        "remote_declaration_sha256",
        "remote_manifest_sha256",
        "inputs",
        "manifest_sha256",
    }:
        _fail()
    if type(value["version"]) is not int or value["version"] != 1:
        _fail()
    records = value["inputs"]
    if not isinstance(records, list) or len(records) > MAX_WHEEL_MANIFEST_ARTIFACTS:
        _fail()
    remote: list[ComfyRegistryWheelArtifact] = []
    local: list[ComfyRegistryReviewedWheelInput] = []
    for record in records:
        if not isinstance(record, dict):
            _fail()
        if record.get("kind") == "remote":
            remote.append(
                ComfyRegistryWheelArtifact(
                    **_entry_fields(
                        record,
                        {field.name for field in fields(ComfyRegistryWheelArtifact)},
                    )
                )
            )
        elif record.get("kind") == "reviewed-local":
            local.append(
                ComfyRegistryReviewedWheelInput(
                    **_entry_fields(
                        record,
                        {field.name for field in fields(ComfyRegistryReviewedWheelInput)},
                    )
                )
            )
        else:
            _fail()
    try:
        remote_manifest = build_comfy_registry_wheel_artifact_manifest(
            _digest(value["remote_declaration_sha256"]),
            _digest(value["target_sha256"]),
            remote,
        )
    except ValueError:
        _fail("invalid_remote_wheel_inputs")
    if not hmac.compare_digest(
        _digest(value["remote_manifest_sha256"]), remote_manifest.manifest_sha256
    ):
        _fail("invalid_remote_wheel_inputs")
    manifest = build_comfy_registry_wheel_input_manifest(
        _digest(value["declaration_sha256"]),
        remote_manifest,
        local,
    )
    if not hmac.compare_digest(_digest(value["manifest_sha256"]), manifest.manifest_sha256):
        _fail("wheel_input_manifest_changed")
    if wheel_input_manifest_payload(manifest) != value:
        _fail()
    return manifest
