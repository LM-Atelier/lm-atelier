"""Content-addressed model binding.

`model_install_ids` names local rows, so it survives an export as a list of
identifiers that mean nothing on the receiving machine. These cover the binding
that does travel, and the fallback that keeps existing installs working.
"""

from __future__ import annotations

import itertools

import local_lm.model_manifests as manifests_module
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.model_manifests import inspect_repository_metadata
from local_lm.model_planner import (
    declared_model_components,
    install_satisfies_components,
    revision_accepts_install,
    revision_declares_a_model,
)
from local_lm.models import ModelComponentManifest, ModelInstall

_COUNTER = itertools.count()


def _unique() -> str:
    return f"case{next(_COUNTER)}"


def _install(session, suffix: str, components: list[tuple[str, str]]) -> ModelInstall:  # type: ignore[no-untyped-def]
    install = ModelInstall(
        id=f"install_{suffix}",
        name=f"acme/model-{suffix}",
        role="image",
        engine="comfyui",
        local_path=f"/models/{suffix}",
        size_bytes=1,
        manifest_json={},
    )
    session.add(install)
    session.flush()
    for folder, digest in components:
        session.add(
            ModelComponentManifest(
                model_install_id=install.id,
                kind="checkpoint",
                relative_path=f"{folder}/weights.safetensors",
                target_folder=folder,
                sha256=digest,
                size_bytes=1,
                required=True,
                metadata_json={},
            )
        )
    session.flush()
    return install


def test_a_workflow_resolves_against_whichever_install_holds_the_files(
    settings: Settings,
) -> None:
    """The portability case: the declared install id is from another machine."""
    digest = "a" * 64
    with SessionLocal() as session:
        local = _install(session, _unique(), [("checkpoints", digest)])
        dependencies = {
            # An id from the machine that exported this workflow.
            "model_install_ids": ["install_from_somewhere_else"],
            "model_components": [{"target_folder": "checkpoints", "sha256": digest}],
        }

        assert revision_accepts_install(session, dependencies, local.id)
        session.rollback()


def test_an_install_missing_a_declared_component_is_refused(settings: Settings) -> None:
    with SessionLocal() as session:
        local = _install(session, _unique(), [("checkpoints", "a" * 64)])
        dependencies = {
            "model_install_ids": [],
            "model_components": [
                {"target_folder": "checkpoints", "sha256": "a" * 64},
                {"target_folder": "vae", "sha256": "b" * 64},
            ],
        }

        # Partial coverage is not coverage.
        assert not install_satisfies_components(
            session, local.id, declared_model_components(dependencies)
        )
        session.rollback()


def test_an_install_with_no_manifest_falls_back_to_the_id_binding(
    settings: Settings,
) -> None:
    """A missing manifest means unknown, not compatible - but must not regress."""
    with SessionLocal() as session:
        local = _install(session, _unique(), [])
        components = [{"target_folder": "checkpoints", "sha256": "a" * 64}]

        assert not install_satisfies_components(session, local.id, components)
        # Content cannot decide, so the id list still does, exactly as before.
        assert revision_accepts_install(
            session,
            {"model_install_ids": [local.id], "model_components": components},
            local.id,
        )
        assert not revision_accepts_install(
            session,
            {"model_install_ids": ["someone-else"], "model_components": components},
            local.id,
        )
        session.rollback()


def test_a_revision_binding_nothing_stays_unconstrained(settings: Settings) -> None:
    with SessionLocal() as session:
        local = _install(session, _unique(), [("checkpoints", "a" * 64)])

        assert not revision_declares_a_model({})
        assert revision_accepts_install(session, {}, local.id)
        assert revision_declares_a_model(
            {"model_components": [{"target_folder": "checkpoints", "sha256": "a" * 64}]}
        )
        session.rollback()


def test_declared_components_are_normalized_and_junk_is_dropped() -> None:
    declared = declared_model_components(
        {
            "model_components": [
                {"target_folder": "vae", "sha256": "B" * 64},
                {"target_folder": "checkpoints", "sha256": "a" * 64},
                # Duplicates collapse; malformed entries are ignored rather than
                # being allowed to widen or narrow the requirement.
                {"target_folder": "vae", "sha256": "b" * 64},
                {"target_folder": "", "sha256": "c" * 64},
                {"target_folder": "loras", "sha256": "too-short"},
                "not-a-mapping",
            ]
        }
    )

    assert declared == [
        {"target_folder": "checkpoints", "sha256": "a" * 64},
        {"target_folder": "vae", "sha256": "b" * 64},
    ]
    assert declared_model_components({}) == []
    assert declared_model_components({"model_components": "nope"}) == []


def _safetensors() -> bytes:
    """The smallest well-formed safetensors file: a length prefix and a header."""
    header = b"{}"
    return len(header).to_bytes(8, "little") + header


def test_the_template_decides_the_component_folder_not_the_repository_layout() -> None:
    """A multi-component model must not land entirely in `checkpoints`.

    `_target_folder` infers the folder from the repository path, which only works
    when the repository happens to name its directories the way ComfyUI does. A
    repository that does not - and most do not - fell back to `checkpoints` for
    every component, so a model whose files really belong in `diffusion_models`,
    `text_encoders` and `vae` recorded three identical wrong folders. Local
    execution still worked, because execution uses the template; only the
    portable record was wrong, which is the kind of defect that surfaces on
    someone else's machine.
    """
    files = {
        "split/dit.safetensors": _safetensors(),
        "split/te.safetensors": _safetensors(),
        "split/vae.safetensors": _safetensors(),
    }
    selected = sorted(files)

    inferred = inspect_repository_metadata(files, selected, role="image")
    assert {item.target_folder for item in inferred.components} == {"checkpoints"}

    declared = inspect_repository_metadata(
        files,
        selected,
        role="image",
        component_folders={
            "split/dit.safetensors": "diffusion_models",
            "split/te.safetensors": "text_encoders",
            "split/vae.safetensors": "vae",
        },
    )

    assert {item.path: item.target_folder for item in declared.components} == {
        "split/dit.safetensors": "diffusion_models",
        "split/te.safetensors": "text_encoders",
        "split/vae.safetensors": "vae",
    }


def test_an_undeclared_component_is_left_exactly_as_it_was() -> None:
    """The template is the authority when it speaks, and silent when it does not.

    Asserted as equivalence rather than against a fixed folder: what matters is
    that naming some components cannot change the classification of the others.
    """
    files = {"loras/style.safetensors": _safetensors()}
    selected = ["loras/style.safetensors"]

    untouched = inspect_repository_metadata(files, selected, role="image")
    with_others_declared = inspect_repository_metadata(
        files,
        selected,
        role="image",
        component_folders={"something/else.safetensors": "vae"},
    )

    assert [item.target_folder for item in with_others_declared.components] == [
        item.target_folder for item in untouched.components
    ]


def test_a_lycoris_adapter_is_recognised_as_a_lora() -> None:
    """LoKr, LoHa and DoRA are LoRAs in every way that matters here.

    ComfyUI loads them through the same loader and providers list them as
    LoRAs, but none of them spells "lora" in a tensor name: LoKr factorises
    into `lokr_w1`/`lokr_w2`, LoHa into `hada_w1_a`/`hada_w2_b`, DoRA adds a
    `dora_scale`. Classified as an unknown blob, a real 1.5 GB adapter finished
    downloading and was then discarded at the contract check for not being what
    it plainly is.
    """
    lokr = [
        "diffusion_model.blocks.0.attn.gate.alpha",
        "diffusion_model.blocks.0.attn.gate.lokr_w1",
        "diffusion_model.blocks.0.attn.gate.lokr_w2",
    ]
    loha = ["diffusion_model.blocks.0.attn.hada_w1_a", "diffusion_model.blocks.0.attn.hada_w2_b"]
    dora = ["diffusion_model.blocks.0.attn.wk.dora_scale"]
    for names in (lokr, loha, dora):
        assert manifests_module._safetensors_kind(names, {}) == "lora"

    # A checkpoint stays a checkpoint: these names mean nothing outside an
    # adapter, so nothing ordinary is promoted by matching them.
    plain = [
        "model.diffusion_model.input_blocks.0.0.weight",
        "model.diffusion_model.output_blocks.0.0.bias",
    ]
    assert manifests_module._safetensors_kind(plain, {}) != "lora"
