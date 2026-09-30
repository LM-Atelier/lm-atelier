"""Provider prose remains bounded and separate from automatic use-case derivation."""

from __future__ import annotations

import pytest

from local_lm.civitai_catalog import CivitaiCatalog
from local_lm.model_manifests import InspectedComponent, ModelManifestInspection
from local_lm.model_planner import ResolvedInstallPlan, resolve_install_plan
from local_lm.profile_use_cases import derive_profile_use_case
from local_lm.provider_descriptions import (
    installed_provider_description,
    merge_provider_descriptions,
    normalize_provider_description,
)


def plan_for_description(description: object) -> ResolvedInstallPlan:
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
                "metadata": {"tags": ["landscape"], "description": description},
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


@pytest.mark.parametrize("value", [None, 17, {}, ["description"], "", " \n", "\ud800"])
def test_invalid_provider_descriptions_are_unavailable(value: object) -> None:
    assert normalize_provider_description(value) == ""


def test_description_bound_and_duplicate_files_do_not_expand_the_snapshot() -> None:
    assert normalize_provider_description("x" * 10_000) == "x" * 8_000
    result = merge_provider_descriptions(["x" * 7_995, "x" * 7_995, "watercolor scenes"])
    assert result == "x" * 7_995 + "\n\nwat"
    assert len(result) == 8_000


def test_descriptions_change_the_plan_without_becoming_a_deterministic_use_case() -> None:
    first = plan_for_description("Watercolor scenes")
    second = plan_for_description("Architectural sketches")
    assert first.plan_hash != second.plan_hash
    assert first.runtime_contract["provider_description"] == "Watercolor scenes"
    assert second.runtime_contract["provider_description"] == "Architectural sketches"
    assert derive_profile_use_case(first.runtime_contract["use_case_metadata"]) == "landscape"
    assert "provider_description" not in plan_for_description(None).runtime_contract


@pytest.mark.parametrize("contract", [{}, {"provider_description": None}, [], "invalid"])
def test_accepted_absent_or_invalid_descriptions_do_not_fall_back_to_live_metadata(
    contract: object,
) -> None:
    assert installed_provider_description(contract, {"description": "Changed live text"}) == ""


def test_accepted_description_wins_over_later_provider_changes() -> None:
    assert (
        installed_provider_description(
            {"provider_description": "Accepted watercolor description"},
            {"description": "Changed live text"},
        )
        == "Accepted watercolor description"
    )
    assert installed_provider_description(None, {"description": "Legacy description"}) == (
        "Legacy description"
    )


def test_civitai_model_and_version_descriptions_are_preserved_once_per_file() -> None:
    files = CivitaiCatalog._normalize_files(
        {"id": 1, "type": "LORA", "description": "Watercolor landscapes"},
        {
            "id": 2,
            "description": "Adds pastel tones",
            "files": [{"id": 3, "name": "a.safetensors"}, {"id": 4, "name": "b.safetensors"}],
        },
    )
    descriptions = [item["metadata"]["description"] for item in files]
    assert descriptions == ["Watercolor landscapes\n\nAdds pastel tones"] * 2
    assert merge_provider_descriptions(descriptions) == descriptions[0]


@pytest.mark.parametrize(
    "description,expected",
    [
        (
            "<p>Watercolor <strong>landscapes</strong>.</p><p>Soft &amp; warm tones.</p>",
            "Watercolor landscapes. Soft & warm tones.",
        ),
        ("<style>color: blue</style><script>neutral()</script><p>Pastel skies</p>", "Pastel skies"),
        ('<p><a href="https://example.test">Ink</a><br>and pencil</p>', "Ink and pencil"),
        ("<p>Watercolor <strong>landscapes", "Watercolor landscapes"),
        ("Paint 2 < 3 panels", "Paint 2 < 3 panels"),
        ('<p title="' + "x" * 9_000 + '">Watercolor</p>', "Watercolor"),
        ("<p>" + "x" * 10_000 + "</p>", "x" * 8_000),
        ('<p title="' + "x" * 32_000 + '">Beyond the input bound</p>', ""),
        ("<![unknown]>", ""),
    ],
    ids=[
        "markup",
        "non-prose",
        "links-and-breaks",
        "unclosed-tags",
        "literal-less-than",
        "long-markup",
        "text-bound",
        "input-bound",
        "malformed-declaration",
    ],
)
def test_civitai_descriptions_keep_bounded_readable_text(description: str, expected: str) -> None:
    files = CivitaiCatalog._normalize_files(
        {"id": 1, "type": "LORA", "description": description},
        {"id": 2, "files": [{"id": 3, "name": "a.safetensors"}]},
    )
    assert files[0]["metadata"]["description"] == expected


def test_accepted_provider_prose_is_not_reinterpreted_after_installation() -> None:
    assert (
        installed_provider_description(
            {"provider_description": "<p>Accepted text</p>"},
            {"description": "New provider prose"},
        )
        == "<p>Accepted text</p>"
    )


@pytest.mark.parametrize("error", [AssertionError, ValueError])
def test_description_parser_failures_discard_partial_text(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    from local_lm.provider_descriptions import _DescriptionText

    def fail_after_text(parser: _DescriptionText, data: str) -> None:
        parser.handle_data(data)
        raise error("Invalid markup")

    monkeypatch.setattr(_DescriptionText, "feed", fail_after_text)
    files = CivitaiCatalog._normalize_files(
        {"id": 1, "type": "LORA", "description": "Watercolor"},
        {"id": 2, "files": [{"id": 3, "name": "a.safetensors"}]},
    )
    assert files[0]["metadata"]["description"] == ""
