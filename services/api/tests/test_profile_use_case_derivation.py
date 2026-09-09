from __future__ import annotations

import pytest
from sqlalchemy import select

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.model_manifests import InspectedComponent, ModelManifestInspection
from local_lm.model_planner import ResolvedInstallPlan, resolve_install_plan
from local_lm.models import ModelInstall, ModelProfile, ModelSource, WorkflowFamily
from local_lm.profile_service import build_profile_for_install, ensure_profile_for_install
from local_lm.workflow_compatibility import ensure_legacy_profile_workflow
from local_lm.workflow_selection import resolve_workflow_family


def _metadata() -> dict[str, object]:
    return {
        "tags": [" coding ", "CODING", "technical\nanswers"],
        "category": "text generation",
        "trained_words": ["structured reply"],
        "base_model": "Neutral base",
        "description": "Do not derive from descriptions",
    }


def _plan(metadata: dict[str, object]) -> ResolvedInstallPlan:
    return resolve_install_plan(
        remote_id="neutral/model",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "model.safetensors",
                "size": 1024,
                "sha256": "b" * 64,
                "metadata": metadata,
            }
        ],
        inspection=ModelManifestInspection(
            architecture="stable-diffusion-xl",
            family="stable-diffusion-xl",
            components=(InspectedComponent("model.safetensors", "checkpoint", "checkpoints"),),
            metadata_files=(),
        ),
        workflow_template_id="neutral-template",
        workflow_template_sha256="c" * 64,
    )


def _install(settings: Settings, metadata: dict[str, object]) -> str:
    with SessionLocal() as session:
        source = ModelSource(
            provider="huggingface",
            remote_id="neutral/profile-model",
            revision="d" * 40,
            metadata_json=metadata,
        )
        session.add(source)
        session.flush()
        install = ModelInstall(
            source_id=source.id,
            name="Neutral chat model",
            role="chat",
            engine="llama.cpp",
            active=True,
            local_path=str(settings.model_dir / "neutral-profile-model"),
            manifest_json={},
        )
        session.add(install)
        session.commit()
        return install.id


def test_install_plan_freezes_use_case_metadata_without_descriptions() -> None:
    plan = _plan(_metadata())
    assert plan.runtime_contract.get("use_case_metadata") == {
        "tags": ["coding", "technical answers"],
        "category": ["text generation"],
        "trained_words": ["structured reply"],
        "base_model": ["Neutral base"],
    }


def test_use_case_metadata_changes_the_immutable_install_plan() -> None:
    first = _plan({"tags": ["landscape"]})
    second = _plan({"tags": ["architecture"]})
    assert first.plan_hash != second.plan_hash


def test_new_install_profile_derives_text_and_mirrors_its_family(settings: Settings) -> None:
    install_id = _install(settings, _metadata())
    with SessionLocal() as session:
        install = session.get(ModelInstall, install_id)
        assert install is not None
        profile = ensure_profile_for_install(session, install)
        assert (
            profile.use_case
            == "coding; technical answers; text generation; structured reply; Neutral base"
        )
        assert getattr(profile, "use_case_derived", False) is True
        family = session.scalar(select(WorkflowFamily).where(WorkflowFamily.name == profile.name))
        assert family is not None and family.use_case == profile.use_case
        other = ModelProfile(
            name="coding technical answers", role="chat", engine="mock", use_case=""
        )
        session.add(other)
        session.flush()
        ensure_legacy_profile_workflow(session, other)
        selected = resolve_workflow_family(
            session,
            capability="chat",
            operation=Operation.TEXT,
            mode="automatic",
            prompt="Help with coding and technical answers",
        )
        assert selected.profile_id == profile.id
        assert selected.workflow_family_id == family.id
        assert {"code", "technical", "answers"}.issubset(selected.matched_terms)
        session.commit()
        session.expire_all()
        persisted = session.get(ModelProfile, profile.id)
        assert persisted is not None and getattr(persisted, "use_case_derived", False) is True
        install.manifest_json = {"use_case_metadata": {"tags": ["Selected revision snapshot"]}}
        provisional = build_profile_for_install(
            install, source_metadata={"tags": ["Later metadata"]}
        )
        assert provisional.use_case == "Selected revision snapshot"
        assert provisional.use_case_derived is True


def test_malformed_provider_metadata_does_not_become_a_use_case(settings: Settings) -> None:
    install_id = _install(
        settings,
        {
            "tags": [None, False, 17, {"text": "ignored"}],
            "category": {"bad": "shape"},
            "base_model": False,
            "trained_words": [],
            "description": "not a use case",
        },
    )
    with SessionLocal() as session:
        install = session.get(ModelInstall, install_id)
        assert install is not None
        profile = ensure_profile_for_install(session, install)
        assert profile.use_case == ""
        assert not getattr(profile, "use_case_derived", False)


def test_derived_profile_text_stays_inside_the_portable_bundle_limit(settings: Settings) -> None:
    install_id = _install(settings, {"tags": [str(i) + "x" * 500 for i in range(40)]})
    with SessionLocal() as session:
        install = session.get(ModelInstall, install_id)
        assert install is not None
        profile = ensure_profile_for_install(session, install)
        assert profile.use_case
        assert len(profile.use_case) <= 1000


@pytest.mark.parametrize("user_text", ["My own description", ""])
def test_existing_profile_text_including_a_manual_clear_is_preserved(
    settings: Settings,
    user_text: str,
) -> None:
    install_id = _install(settings, _metadata())
    with SessionLocal() as session:
        profile = ModelProfile(
            name="Existing profile",
            use_case=user_text,
            role="chat",
            engine="llama.cpp",
            model_install_id=install_id,
        )
        session.add(profile)
        session.commit()
        install = session.get(ModelInstall, install_id)
        assert install is not None
        same = ensure_profile_for_install(session, install)
        assert same.id == profile.id and same.use_case == user_text
        assert not getattr(same, "use_case_derived", False)
