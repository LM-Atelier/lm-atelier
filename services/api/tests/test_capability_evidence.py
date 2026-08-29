from __future__ import annotations

import itertools

from local_lm.adapters.contracts import ADAPTER_CONTRACT_VERSION
from local_lm.capability_evidence import ACTIVATION_ARTIFACT_KEY, current_capability_evidence
from local_lm.comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.hardware import hardware_capability_class
from local_lm.model_planner import (
    ACTIVATION_PROBE_VERSION,
    LAUNCH_CONTRACT_VERSION,
    media_workflow_contract_version,
)
from local_lm.models import (
    ModelCapabilityEvidence,
    ModelInstall,
    WorkflowDefinition,
    WorkflowRevision,
)

_TEMPLATE_SHA = "c" * 64
_COUNTER = itertools.count()


def _unique() -> str:
    return f"case{next(_COUNTER)}"


def _install(suffix: str) -> ModelInstall:
    """Unique per test; the database is shared across this module."""
    return ModelInstall(
        id=f"model_media_{suffix}",
        name="Media",
        role="image",
        engine="comfyui",
        local_path="C:/synthetic/media",
        manifest_json={
            "files": ["model.safetensors"],
            "expected_sha256": {"model.safetensors": "d" * 64},
            "workflow_template_id": "template_image",
            "workflow_template_sha256": _TEMPLATE_SHA,
        },
        active=True,
    )


def _add_revision(session, identifier: str, operation: str, artifact: str, install_id: str) -> None:  # type: ignore[no-untyped-def]
    """Insert a definition and its revision, which reference each other."""
    definition = WorkflowDefinition(
        id=f"wf_{identifier}", name=identifier, description="", operation=operation
    )
    revision = WorkflowRevision(
        id=f"rev_{identifier}",
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        api_graph_json={"1": {"class_type": "Synthetic"}},
        dependencies_json={
            "model_install_ids": [install_id],
            "compiler_version": COMFY_TEMPLATE_COMPILER_VERSION,
            "template_id": "template_image",
            "template_sha256": _TEMPLATE_SHA,
        },
        artifact_sha256=artifact,
        trusted=True,
    )
    session.add(definition)
    session.flush()
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()


def _evidence(
    settings: Settings, install: ModelInstall, contract: str | None
) -> ModelCapabilityEvidence:
    return ModelCapabilityEvidence(
        model_install_id=install.id,
        evidence_key=f"{install.id:e<64}"[:64],
        result="ready",
        component_hashes_json=dict(install.manifest_json["expected_sha256"]),
        runtime_build="comfy-test",
        adapter_contract_version=ADAPTER_CONTRACT_VERSION,
        launch_contract_version=LAUNCH_CONTRACT_VERSION,
        workflow_contract_version=contract,
        hardware_class=hardware_capability_class(settings),
        probe_version=ACTIVATION_PROBE_VERSION,
        details_json={},
    )


def test_evidence_survives_a_compiler_change_that_alters_nothing(settings: Settings) -> None:
    """The whole point: a compiler bump no longer demotes a proven model."""
    install = _install(_unique())
    install.manifest_json = {**install.manifest_json, ACTIVATION_ARTIFACT_KEY: "a" * 64}
    with SessionLocal() as session:
        session.add(install)
        session.flush()
        _add_revision(session, f"primary_{install.id}", "text_to_image", "a" * 64, install.id)
        session.add(_evidence(settings, install, "a" * 64))
        session.commit()
        assert current_capability_evidence(session, install, settings, None)


def test_a_derived_edit_workflow_does_not_invalidate_creation_readiness(
    settings: Settings,
) -> None:
    """An install can hold a primary workflow and derived edit workflows.

    Comparing against whichever revision ran last would let an edit demote
    creation readiness while both workflows are perfectly healthy.
    """
    install = _install(_unique())
    install.manifest_json = {**install.manifest_json, ACTIVATION_ARTIFACT_KEY: "a" * 64}
    with SessionLocal() as session:
        session.add(install)
        session.flush()
        _add_revision(session, f"primary_{install.id}", "text_to_image", "a" * 64, install.id)
        _add_revision(session, f"edit_{install.id}", "image_to_image", "b" * 64, install.id)
        session.add(_evidence(settings, install, "a" * 64))
        session.commit()
        assert current_capability_evidence(session, install, settings, None)


def test_legacy_evidence_is_accepted_for_its_own_recorded_compiler(
    settings: Settings,
) -> None:
    """Existing installations keep evidence they already earned."""
    install = _install(_unique())
    with SessionLocal() as session:
        session.add(install)
        session.flush()
        _add_revision(session, f"primary_{install.id}", "text_to_image", "a" * 64, install.id)
        session.add(
            _evidence(
                settings,
                install,
                media_workflow_contract_version(_TEMPLATE_SHA, COMFY_TEMPLATE_COMPILER_VERSION),
            )
        )
        session.commit()
        assert current_capability_evidence(session, install, settings, None)


def test_legacy_evidence_from_another_compiler_is_refused(settings: Settings) -> None:
    """The compatibility path is bounded so it cannot validate anything else."""
    install = _install(_unique())
    with SessionLocal() as session:
        session.add(install)
        session.flush()
        _add_revision(session, f"primary_{install.id}", "text_to_image", "a" * 64, install.id)
        session.add(
            _evidence(
                settings,
                install,
                media_workflow_contract_version(
                    _TEMPLATE_SHA, COMFY_TEMPLATE_COMPILER_VERSION + 99
                ),
            )
        )
        session.commit()
        assert current_capability_evidence(session, install, settings, None) is None


def test_evidence_for_a_different_artifact_is_refused(settings: Settings) -> None:
    install = _install(_unique())
    install.manifest_json = {**install.manifest_json, ACTIVATION_ARTIFACT_KEY: "a" * 64}
    with SessionLocal() as session:
        session.add(install)
        session.flush()
        _add_revision(session, f"primary_{install.id}", "text_to_image", "a" * 64, install.id)
        session.add(_evidence(settings, install, "f" * 64))
        session.commit()
        assert current_capability_evidence(session, install, settings, None) is None
