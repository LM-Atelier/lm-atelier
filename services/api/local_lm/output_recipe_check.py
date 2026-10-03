"""Compare a portable generation record with what this installation holds.

The check reads the record and this computer's own rows and nothing else. It
never downloads, installs, trusts, activates or runs anything, and it keeps no
copy of the record. Every requirement is matched by its exact content identity -
a file hash or a workflow's recomputed identity - never by a name, because two
files called the same thing are not the same file.

The answer names each requirement by the hash the record gave and says whether
it is here and ready, here but not ready, or absent. It repeats nothing else from
the record, so a prompt inside it is never echoed back.

The file may also be a bundle saved with its picture. Then the bundle is read
whole and strictly first - a bundle that is not exactly as written is refused -
and the answer also gives the copy's size and its hash beside the hash of the
output it was copied from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    Artifact,
    ModelAssetInstall,
    ModelComponentManifest,
    ModelInstall,
    WorkflowRevision,
)
from .output_recipe_bundle import (
    MAX_BUNDLE_BYTES,
    OutputRecipeBundleFormatError,
    open_output_recipe_bundle,
)
from .output_recipe_v1 import MAX_RECORD_BYTES, OutputRecipeFormatError, open_output_recipe

RequirementState = Literal["present", "inactive", "missing"]

#: What a requirement is, in the order a person would set one up.
REQUIREMENT_KINDS: Final = ("workflow", "model_file", "lora", "input")
#: The largest file the check reads: a bundle at its largest. A bare record is
#: held to its own, far smaller, bound.
MAX_CHECK_BYTES: Final = MAX_BUNDLE_BYTES
BUNDLE_SIGNATURE: Final = b"PK\x03\x04"


def most_read_for(prefix: bytes) -> int:
    """The most of a file the check reads, judged by how the file begins.

    A file that does not begin as a bundle can only be a record, so it is held
    to a record's bound rather than read on to a bundle's.
    """

    return MAX_CHECK_BYTES if prefix.startswith(BUNDLE_SIGNATURE) else MAX_RECORD_BYTES


class OutputRecipeCheckRefused(Exception):
    """A record that cannot be checked, as a code and a sentence that echo nothing."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RecordFile:
    """A record read strictly, and the bundle it came in when it came in one."""

    record: dict[str, Any]
    bundle: dict[str, Any] | None


def read_record_file(content: bytes) -> RecordFile:
    """Read a bare record, or the record inside a bundle saved with its picture."""

    if not content.startswith(BUNDLE_SIGNATURE):
        if len(content) > MAX_RECORD_BYTES:
            raise _too_large()
        try:
            return RecordFile(open_output_recipe(content), None)
        except OutputRecipeFormatError as exc:
            raise OutputRecipeCheckRefused(
                422,
                "output-recipe-unreadable",
                "This file is not a generation record this version can read.",
            ) from exc
    if len(content) > MAX_CHECK_BYTES:
        raise _too_large()
    try:
        bundle = open_output_recipe_bundle(content)
    except OutputRecipeBundleFormatError as exc:
        raise OutputRecipeCheckRefused(
            422,
            "output-recipe-bundle-unreadable",
            "This file is not a generation record bundle this version can read.",
        ) from exc
    return RecordFile(bundle["record"], bundle)


def check_output_recipe_file(session: Session, content: bytes) -> dict[str, Any]:
    """Check a bare record, or the record inside a bundle saved with its picture."""

    read = read_record_file(content)
    if read.bundle is None:
        return {**_report(session, read.record), "picture": None, "bundled_inputs": []}
    picture = read.bundle["manifest"]["picture"]
    return {
        **_report(session, read.record),
        "picture": {
            "sha256": picture["sha256"],
            "copy_of": picture["copy_of"],
            "width": read.bundle["width"],
            "height": read.bundle["height"],
        },
        # The recorded inputs the bundle carries a copy of, by position, which a
        # new version can use in place of ones that are not here.
        "bundled_inputs": [item["position"] for item in read.bundle["inputs"]],
    }


def check_output_recipe(session: Session, content: bytes) -> dict[str, Any]:
    """Say which of a bare record's requirements this installation holds, by exact identity."""

    if content.startswith(BUNDLE_SIGNATURE):
        return check_output_recipe_file(session, content)
    return _report(session, read_record_file(content).record)


def _too_large() -> OutputRecipeCheckRefused:
    return OutputRecipeCheckRefused(
        413, "output-recipe-too-large", "This file is larger than a generation record can be."
    )


def _report(session: Session, record: dict[str, Any]) -> dict[str, Any]:
    requirements: list[dict[str, Any]] = []
    workflow = record["workflow"]
    if workflow is not None:
        requirements.append(
            {
                "kind": "workflow",
                "sha256": workflow["artifact_sha256"],
                "role": None,
                "state": _workflow_state(session, workflow["artifact_sha256"]),
            }
        )
    model = record["model"]
    if model is not None:
        for name in sorted(model["files"]):
            digest = model["files"][name]
            requirements.append(
                {
                    "kind": "model_file",
                    "sha256": digest,
                    "role": None,
                    "state": _model_file_state(session, digest),
                }
            )
    for lora in record["loras"]:
        requirements.append(
            {
                "kind": "lora",
                "sha256": lora["sha256"],
                "role": None,
                "state": _lora_state(session, lora["sha256"]),
            }
        )
    for item in record["inputs"]:
        requirements.append(
            {
                "kind": "input",
                "sha256": item["sha256"],
                "role": item["role"],
                "state": _input_state(session, item["sha256"]),
            }
        )
    return {
        "digest": record["digest"],
        "operation": record["operation"],
        "requirements": requirements,
        "all_present": all(item["state"] == "present" for item in requirements),
        "reproducibility": record["reproducibility"],
        "not_recorded": record["not_recorded"],
    }


def _workflow_state(session: Session, identity: str) -> RequirementState:
    """A workflow is ready when a trusted revision executes exactly this graph."""

    trusted = session.scalars(
        select(WorkflowRevision.trusted).where(WorkflowRevision.artifact_sha256 == identity)
    ).all()
    if not trusted:
        return "missing"
    return "present" if any(trusted) else "inactive"


def _model_file_state(session: Session, digest: str) -> RequirementState:
    """A model file is ready when an active install holds a file with this hash."""

    holders = list(
        session.scalars(
            select(ModelInstall.active)
            .join(
                ModelComponentManifest, ModelComponentManifest.model_install_id == ModelInstall.id
            )
            .where(ModelComponentManifest.sha256 == digest)
        ).all()
    )
    # Installs made before component manifests existed record their hashes in
    # the install's own manifest instead.
    for install in session.scalars(select(ModelInstall)).all():
        expected = (install.manifest_json or {}).get("expected_sha256")
        if isinstance(expected, dict) and digest in expected.values():
            holders.append(install.active)
    if not holders:
        return "missing"
    return "present" if any(holders) else "inactive"


def _lora_state(session: Session, digest: str) -> RequirementState:
    """A LoRA is ready when an active asset holds a file with this hash."""

    holders = [
        asset.active
        for asset in session.scalars(select(ModelAssetInstall)).all()
        if (asset.manifest_json or {}).get("sha256") == digest
    ]
    if not holders:
        return "missing"
    return "present" if any(holders) else "inactive"


def _input_state(session: Session, digest: str) -> RequirementState:
    return "present" if session.get(Artifact, f"sha256:{digest}") is not None else "missing"
