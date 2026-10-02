"""Compare a portable generation record with what this installation holds.

The check reads the record and this computer's own rows and nothing else. It
never downloads, installs, trusts, activates or runs anything, and it keeps no
copy of the record. Every requirement is matched by its exact content identity -
a file hash or a workflow's recomputed identity - never by a name, because two
files called the same thing are not the same file.

The answer names each requirement by the hash the record gave and says whether
it is here and ready, here but not ready, or absent. It repeats nothing else from
the record, so a prompt inside it is never echoed back.
"""

from __future__ import annotations

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
from .output_recipe_v1 import MAX_RECORD_BYTES, OutputRecipeFormatError, open_output_recipe

RequirementState = Literal["present", "inactive", "missing"]

#: What a requirement is, in the order a person would set one up.
REQUIREMENT_KINDS: Final = ("workflow", "model_file", "lora", "input")


class OutputRecipeCheckRefused(Exception):
    """A record that cannot be checked, as a code and a sentence that echo nothing."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def check_output_recipe(session: Session, content: bytes) -> dict[str, Any]:
    """Say which of a record's requirements this installation holds, by exact identity."""

    if len(content) > MAX_RECORD_BYTES:
        raise OutputRecipeCheckRefused(
            413, "output-recipe-too-large", "This file is larger than a generation record can be."
        )
    try:
        record = open_output_recipe(content)
    except OutputRecipeFormatError as exc:
        raise OutputRecipeCheckRefused(
            422,
            "output-recipe-unreadable",
            "This file is not a generation record this version can read.",
        ) from exc

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
