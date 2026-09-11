"""A workflow that never reads the description is refused before it runs."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from local_lm import prompt_binding
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ModelInstall,
    ModelProfile,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.prompt_binding import (
    binds_prompt,
    frozen_prompt,
    ignores_the_description,
    substitutable_keys,
)

pytestmark = pytest.mark.asyncio


#: What a workflow declaring nothing of its own offers. Computed rather than
#: written out, so a change to the built-in settings cannot silently leave
#: these tests asserting against a set the product no longer has.
_KEYS = substitutable_keys("text_to_image", None)


def _graph(prompt_value: object) -> dict[str, Any]:
    return {
        "encode": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_value}},
        "sampler": {"class_type": "KSampler", "inputs": {"positive": ["encode", 0]}},
    }


def test_a_graph_naming_the_placeholder_binds_the_description() -> None:
    assert binds_prompt(_graph("${prompt}")) is True


def test_a_graph_with_a_baked_in_prompt_does_not() -> None:
    assert binds_prompt(_graph("a post-rain dusk metropolis, anamorphic lens")) is False


def test_an_embedded_placeholder_is_not_a_binding() -> None:
    """The compiler substitutes a whole value or nothing.

    `ComfyUIAdapter._compile` takes `value[2:-1]` off a string it has required
    to begin `${` and end `}`, so `masterpiece, ${prompt}` looks for a
    parameter named `masterpiece, ${prompt` and, finding none, leaves the text
    exactly as written. Reading an embedded placeholder as a binding would let
    this check pass for a graph the description still cannot reach.
    """
    assert binds_prompt(_graph("masterpiece, ${prompt}")) is False
    assert binds_prompt(_graph("${prompt} in ink")) is False


def test_the_placeholder_is_found_wherever_it_sits() -> None:
    assert binds_prompt({"a": {"b": [{"c": ["${prompt}"]}]}}) is True
    assert binds_prompt({"a": [1, None, True, {"b": 2}]}) is False


def test_an_empty_graph_binds_nothing() -> None:
    assert binds_prompt({}) is False


def test_a_node_holding_words_of_its_own_has_a_frozen_prompt() -> None:
    assert frozen_prompt(_graph("a post-rain dusk metropolis"), _KEYS) is True
    assert frozen_prompt({"n": {"inputs": {"prompt": "an action movie trailer"}}}, _KEYS) is True


def test_a_graph_with_no_prompt_input_at_all_freezes_nothing() -> None:
    """A fragment is not a workflow that discards what you typed.

    These are the shapes the repository is full of - a lone loader, a single
    custom node, a stub standing in for a graph while some other mechanism is
    exercised. None of them offers to take a description, so none of them can
    be said to ignore one.
    """
    assert (
        frozen_prompt({"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "x"}}}, _KEYS)
        is False
    )
    assert (
        frozen_prompt(
            {"1": {"class_type": "ConstructedSize", "inputs": {"aspect_ratio": "portrait"}}}, _KEYS
        )
        is False
    )
    assert frozen_prompt({"node": {"class_type": "TestOutput"}}, _KEYS) is False
    assert frozen_prompt({}, _KEYS) is False


def test_an_empty_or_placeholder_value_is_not_words_of_its_own() -> None:
    assert frozen_prompt(_graph(""), _KEYS) is False
    assert frozen_prompt(_graph("   "), _KEYS) is False
    assert frozen_prompt(_graph("${prompt}"), _KEYS) is False
    # A binding to some other input the compiler supplies is still a binding.
    assert frozen_prompt(_graph("${negative_prompt}"), _KEYS) is False


@pytest.mark.parametrize("operation", ["text_to_image", "text_to_video"])
def test_a_description_only_operation_that_reads_no_description_is_ignored_input(
    operation: str,
) -> None:
    assert ignores_the_description("comfyui", operation, _graph("baked in"), None) is True
    assert ignores_the_description("comfyui", operation, _graph("${prompt}"), None) is False


@pytest.mark.parametrize("operation", ["text_to_image", "text_to_video"])
def test_a_graph_that_offers_no_prompt_is_left_alone(operation: str) -> None:
    """The narrowing that keeps this off every stub in the repository."""
    fragment = {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "x"}}}
    assert ignores_the_description("comfyui", operation, fragment, None) is False


def test_a_bound_positive_prompt_makes_a_fixed_negative_one_ordinary() -> None:
    """Hard-coding the negative prompt beside a bound positive one is normal."""
    graph = {
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": "${prompt}"}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality"}},
    }

    assert frozen_prompt(graph, _KEYS) is True
    assert ignores_the_description("comfyui", "text_to_image", graph, None) is False


@pytest.mark.parametrize("operation", ["image_to_image", "image_to_video"])
def test_an_operation_with_a_source_picture_is_left_alone(operation: str) -> None:
    """An upscale or a restoration works from the picture and needs no words."""
    assert ignores_the_description("comfyui", operation, _graph("baked in"), None) is False


def test_another_engine_binds_its_inputs_its_own_way() -> None:
    """The mock engine stands in for a backend rather than compiling a graph."""
    assert ignores_the_description("mock", "text_to_image", _graph("baked in"), None) is False


def _comfy_image_profile(session: Session) -> None:
    """Point the default image profile at an installed ComfyUI model.

    Selection refuses an unavailable profile before it ever looks at the
    revision, so without this the checks below would pass for the wrong reason.
    """
    install = ModelInstall(
        name="Installed image model",
        role="image",
        engine="comfyui",
        local_path="C:/managed/installed-image-model",
        manifest_json={},
        active=True,
    )
    session.add(install)
    session.flush()
    profile = session.query(ModelProfile).filter_by(role="image", is_default=True).one()
    profile.engine = "comfyui"
    profile.model_install_id = install.id


def _comfy_family(
    session: Session,
    *,
    name: str,
    prompt_value: object,
    operation: str = "text_to_image",
    input_schema: dict[str, Any] | None = None,
) -> tuple[str, str]:
    family = WorkflowFamily(name=name)
    definition = WorkflowDefinition(
        family=family,
        variant_key="create",
        name=f"{name} create",
        operation=operation,
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="comfyui",
        api_graph_json=_graph(prompt_value),
        input_schema_json=input_schema or {"type": "object", "properties": {}},
        dependencies_json={},
        trusted=True,
    )
    preference = WorkflowPreference(family=family, selector_capability="image")
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return family.id, revision.id


async def test_the_catalog_reports_a_workflow_that_ignores_the_description_unavailable(
    client: AsyncClient, settings: Settings
) -> None:
    """The tool says so before anyone picks it.

    Readiness is what decides whether a workflow is offered at all, so a
    workflow the description cannot reach belongs in the same unavailable list
    as one built for another engine.
    """
    settings.media_engine = "comfyui"
    with SessionLocal() as session:
        _comfy_image_profile(session)
        ignored, _ = _comfy_family(
            session, name="Baked-in cinematic", prompt_value="a post-rain dusk metropolis"
        )
        reads, _ = _comfy_family(session, name="Reads the description", prompt_value="${prompt}")
        session.commit()

    response = await client.get("/api/workflow-families?selector_capability=image")

    assert response.status_code == 200, response.text
    cards = {item["id"]: item for item in response.json()}
    refused = cards[ignored]["variants"][0]
    assert refused["readiness"] == "unavailable"
    assert refused["readiness_reason"] == "revision_ignores_the_description"
    # The control. Identical in every respect but the one value the check
    # reads, so a refusal that fired on the engine, the operation or the empty
    # schema would fail here rather than pass quietly.
    accepted = cards[reads]["variants"][0]
    assert accepted["readiness_reason"] != "revision_ignores_the_description"


async def test_a_turn_cannot_select_a_workflow_that_ignores_the_description(
    client: AsyncClient, settings: Settings
) -> None:
    """The refusal is on the turn, not only on a helper.

    A helper that answers correctly while nothing calls it leaves the defect
    exactly where it was: the turn accepted, the description shown in the
    conversation beside a picture made without it. This drives the real
    endpoint so that removing the call from selection fails here. Selection
    refuses before any run is created, so nothing is dispatched.
    """
    settings.media_engine = "comfyui"
    with SessionLocal() as session:
        _comfy_image_profile(session)
        family_id, _ = _comfy_family(
            session, name="Baked-in cinematic turn", prompt_value="a post-rain dusk metropolis"
        )
        session.commit()

    chat = (await client.post("/api/chats", json={"title": "Ignored description"})).json()
    await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "a slow aerial drift over an autumn forest canopy at sunrise",
            "mode": "image",
        },
    )

    assert response.status_code >= 400, response.text
    assert "ignores_the_description" in response.text


async def test_a_project_pin_on_such_a_workflow_is_refused_too(
    client: AsyncClient, settings: Settings
) -> None:
    """The pin branch reaches execution by its own road.

    Its own docstring records that it once made only the first of the checks
    the other branches make, so a pin for another engine was selected and then
    failed during execution with an error that never mentioned the pin. Adding
    a check to selection and readiness and not to this one would rebuild
    exactly that hole, so this drives a pinned turn rather than reading the
    message table.
    """
    settings.media_engine = "comfyui"
    with SessionLocal() as session:
        _comfy_image_profile(session)
        _, revision_id = _comfy_family(
            session, name="Baked-in cinematic pin", prompt_value="a post-rain dusk metropolis"
        )
        session.commit()

    project = (await client.post("/api/projects", json={"name": "Pinned"})).json()
    pinned = await client.put(
        f"/api/projects/{project['id']}/workflow-selections/image",
        json={"mode": "revision", "workflow_revision_id": revision_id},
    )
    assert pinned.status_code == 200, pinned.text
    chat = (
        await client.post(
            "/api/chats", json={"title": "Pinned description", "project_id": project["id"]}
        )
    ).json()

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "a slow aerial drift over an autumn forest canopy at sunrise",
            "mode": "image",
        },
    )

    assert response.status_code >= 400, response.text
    assert "never reads what you type" in response.text


def test_a_compound_placeholder_is_not_a_binding() -> None:
    """The shape opens and closes like a placeholder and is not one.

    `ComfyUIAdapter._compile` looks up the WHOLE inner text as a parameter name,
    so `${prompt}${negative_prompt}` asks for a key called
    `prompt}${negative_prompt`, finds none, and sends the braces to the model
    verbatim. Measured against the real compiler, these four lose the
    description; the shipped shape test let every one of them through.
    """
    for value in (
        "${prompt}${negative_prompt}",
        "${prompt} ${negative_prompt}",
        "${prompt}${prompt}",
        "${negative_prompt}${prompt}",
    ):
        assert binds_prompt(_graph(value)) is False, value
        assert frozen_prompt(_graph(value), _KEYS) is True, value
        assert ignores_the_description("comfyui", "text_to_image", _graph(value), None) is True, (
            value
        )


def test_a_declared_setting_is_a_real_binding() -> None:
    """The control that keeps this from refusing working workflows.

    A workflow may route its prompt through its own setting. That value is not
    words of the author's own - the compiler replaces it - so the workflow must
    not be refused. This is why the rule asks for the NAME rather than the shape.
    """
    schema = {"type": "object", "properties": {"house_style": {"type": "string", "default": "ink"}}}
    keys = substitutable_keys("text_to_image", schema)
    assert keys is not None and "house_style" in keys

    assert frozen_prompt(_graph("${house_style}"), keys) is False
    assert (
        ignores_the_description("comfyui", "text_to_image", _graph("${house_style}"), schema)
        is False
    )
    # And without that declaration the same value names nothing the compiler
    # supplies, so it reaches the model as literal braces.
    assert (
        ignores_the_description("comfyui", "text_to_image", _graph("${house_style}"), None) is True
    )


def test_a_key_no_pattern_would_allow_is_still_a_binding() -> None:
    """Why a stricter regex could not have been the fix.

    A workflow setting key may be any printable string, so a workflow can
    legally declare a key named exactly like a concatenation. Measured: the
    compiler then substitutes it. A shape-based rule would refuse that workflow,
    which is the wrong-refusal this rule exists to avoid.
    """
    name = "prompt}${negative_prompt"
    schema = {"type": "object", "properties": {name: {"type": "string", "default": "chosen"}}}
    keys = substitutable_keys("text_to_image", schema)

    assert keys is not None and name in keys
    assert frozen_prompt(_graph("${" + name + "}"), keys) is False


def test_an_unreadable_schema_makes_no_claim_and_does_not_raise() -> None:
    """This runs while listing workflows, so it must not be able to break that.

    Nothing else on the listing path parses a stored input schema, so this call
    is the first. A schema the registry refuses must degrade to the shipped
    behaviour - no names known, fall back to the shape - rather than escaping
    and making every workflow unlistable.
    """
    for schema in (
        ["not", "a", "mapping"],
        "not a mapping",
        {"type": "object", "properties": {"prompt": {"default": "reserved"}}},
        {"type": "object", "properties": {"x" * 300: {"default": 1}}},
        {"type": "object", "properties": {}, "x-lm-atelier-video-length": "garbage"},
    ):
        assert substitutable_keys("text_to_image", schema) is None
        # Falls back to the shipped shape test, which still refuses plain words.
        assert (
            ignores_the_description("comfyui", "text_to_image", _graph("baked in"), schema) is True
        )
        assert (
            ignores_the_description("comfyui", "text_to_image", _graph("${anything}"), schema)
            is False
        )


def test_the_key_set_carries_what_the_adapter_supplies() -> None:
    """Some names never appear in a schema because the adapter adds them.

    Asked of both kinds of work on purpose. `prompt` and `negative_prompt` are
    also image settings, so for an image workflow the registry would supply them
    whether or not they were named here, and an assertion made only there would
    pass over their removal. A video workflow has neither, so it is the one that
    tells the two sources apart.
    """
    for operation in ("text_to_image", "text_to_video"):
        role_keys = substitutable_keys(operation, None)
        assert role_keys is not None
        for name in ("prompt", "negative_prompt", "mask", "input_image", "input_images", "loras"):
            assert name in role_keys, (operation, name)

    keys = substitutable_keys("text_to_image", None)
    assert keys is not None
    for name in ("prompt", "negative_prompt", "mask", "input_image", "input_images"):
        assert name in keys, name
    # Numbered attachments are a family rather than a list, matched by pattern.
    # The pattern has to match the WHOLE name and has to accept more than one
    # digit: a prefix match would treat `${input_image_7_style}` as a binding
    # when nothing supplies it, and a single-digit one would refuse a workflow
    # that reads its twelfth attachment.
    assert frozen_prompt(_graph("${input_image_7}"), keys) is False
    assert frozen_prompt(_graph("${input_image_12}"), keys) is False
    assert frozen_prompt(_graph("${input_image_7_style}"), keys) is True
    assert frozen_prompt(_graph("${xinput_image_7}"), keys) is True
    # The adapter writes these keys in lower case, so the match is case
    # sensitive on purpose; a capitalised name is nobody's parameter.
    assert frozen_prompt(_graph("${INPUT_IMAGE_7}"), keys) is True


def test_a_video_workflow_reads_the_video_settings() -> None:
    """The two roles declare different controls, and the set must follow."""
    image = substitutable_keys("text_to_image", None)
    video = substitutable_keys("text_to_video", None)
    assert image is not None and video is not None
    assert "frames" in video and "frames" not in image


def _length_contract(frames: str = "frames", fps: str = "fps") -> dict[str, Any]:
    """A video workflow that offers a duration instead of a frame count.

    Built to the contract's own rules rather than sketched: the two named
    properties have to exist, the frame window has to sit on the declared
    alignment, and the frame rate property has to agree with the rational one.
    """
    return {
        "type": "object",
        "properties": {
            frames: {"type": "integer", "minimum": 16, "maximum": 160, "default": 16},
            fps: {"type": "number", "default": 16.0},
        },
        "x-lm-atelier-video-length": {
            "version": 1,
            "frames_parameter": frames,
            "fps_parameter": fps,
            "fps_numerator": 16,
            "fps_denominator": 1,
            "frame_alignment": 16,
            "frame_offset": 0,
        },
    }


def test_a_length_contract_does_not_take_away_its_own_bindings() -> None:
    """The two names a length contract hides are still bound at dispatch.

    Declaring a duration REPLACES the frame count and the frame rate in the
    settings a caller may set, so neither is among the workflow's fields. The
    run writes both back under the contract's own names before the graph is
    compiled, so a graph binding one is binding something real, and refusing it
    would make a working video workflow unusable.
    """
    schema = _length_contract()
    keys = substitutable_keys("text_to_video", schema)
    assert keys is not None
    assert {"frames", "fps"} <= keys
    assert "duration_seconds" in keys

    for value in ("${frames}", "${fps}"):
        assert frozen_prompt(_graph(value), keys) is False, value
        assert (
            ignores_the_description("comfyui", "text_to_video", _graph(value), schema) is False
        ), value


def test_a_contract_may_only_rename_an_engine_setting() -> None:
    """The names come from the contract, but the contract cannot invent them.

    A length contract must point at settings the engine already has, so the two
    names it hides are always names the run knows how to supply. A contract
    naming anything else is not readable at all, and the rule makes no claim
    about names rather than guessing.
    """
    invented = _length_contract(frames="length_in_frames", fps="frame_rate")

    assert substitutable_keys("text_to_video", invented) is None
    assert ignores_the_description("comfyui", "text_to_video", _graph("baked in"), invented) is True


def test_a_lora_selection_is_supplied_beside_the_declared_settings() -> None:
    """Not every name in the parameter map comes from the schema or the adapter.

    A LoRA selection is written into the run's settings by the orchestrator, so
    it reaches the compiler as an ordinary parameter even on a workflow whose
    schema never mentions it.
    """
    keys = substitutable_keys("text_to_image", None)
    assert keys is not None and "loras" in keys
    assert frozen_prompt(_graph("${loras}"), keys) is False


def test_an_unreadable_length_contract_cannot_empty_the_catalog() -> None:
    """The failure nobody would have thought to name.

    A frame rate stored as an integer too large to become a float raises
    OverflowError out of the length contract, which is neither a value error nor
    a lookup and so would have escaped any list of expected failures. This call
    runs while the workflow list is built, so an escape here empties the whole
    catalog rather than refusing one workflow.
    """
    schema = _length_contract()
    schema["properties"]["fps"]["default"] = 10**400

    assert substitutable_keys("text_to_video", schema) is None
    assert ignores_the_description("comfyui", "text_to_video", _graph("baked in"), schema) is True
    assert ignores_the_description("comfyui", "text_to_video", _graph("${prompt}"), schema) is False


#: A workflow that routes the description through a setting of its own. Its
#: graph holds nothing but a placeholder, so only the declared name says whether
#: the compiler would replace it - which makes it the one shape that can tell a
#: call site passing the schema apart from one passing nothing.
_DECLARES_ITS_OWN = {
    "type": "object",
    "properties": {"house_style": {"type": "string", "default": "art nouveau"}},
}


async def test_the_catalog_offers_a_workflow_that_declares_its_own_prompt_setting(
    client: AsyncClient, settings: Settings
) -> None:
    """The readiness path must ask the question with the workflow's own schema.

    Passing nothing where the schema belongs would refuse this workflow, and the
    person would be told a workflow that works never reads what they type.
    """
    settings.media_engine = "comfyui"
    with SessionLocal() as session:
        _comfy_image_profile(session)
        declared, _ = _comfy_family(
            session,
            name="Routes the description through a style",
            prompt_value="${house_style}",
            input_schema=_DECLARES_ITS_OWN,
        )
        undeclared, _ = _comfy_family(
            session, name="Names a setting it never declares", prompt_value="${house_style}"
        )
        session.commit()

    response = await client.get("/api/workflow-families?selector_capability=image")

    assert response.status_code == 200, response.text
    cards = {item["id"]: item for item in response.json()}
    offered = cards[declared]["variants"][0]
    assert offered["readiness_reason"] != "revision_ignores_the_description"
    # The same graph without the declaration. Identical but for the schema, so
    # a readiness path that ignored the schema would treat both alike.
    refused = cards[undeclared]["variants"][0]
    assert refused["readiness"] == "unavailable"
    assert refused["readiness_reason"] == "revision_ignores_the_description"


async def test_a_turn_may_select_a_workflow_that_declares_its_own_prompt_setting(
    client: AsyncClient, settings: Settings
) -> None:
    """Selection must ask with the schema too, or it refuses a working workflow."""
    settings.media_engine = "comfyui"
    with SessionLocal() as session:
        _comfy_image_profile(session)
        family_id, _ = _comfy_family(
            session,
            name="Style-routed turn",
            prompt_value="${house_style}",
            input_schema=_DECLARES_ITS_OWN,
        )
        session.commit()

    chat = (await client.post("/api/chats", json={"title": "Style routed"})).json()
    selected = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )

    assert selected.status_code == 200, selected.text
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "a harbour at first light", "mode": "image"},
    )

    assert "ignores_the_description" not in response.text


async def test_a_pinned_workflow_that_declares_its_own_prompt_setting_still_runs(
    client: AsyncClient, settings: Settings
) -> None:
    """And so must a project pin, which is the third place the question is asked.

    A pin is resolved when a turn uses it rather than when it is set, so the
    refusal shows up there. Both revisions below are pinned successfully; only
    the turn tells them apart.
    """
    settings.media_engine = "comfyui"
    with SessionLocal() as session:
        _comfy_image_profile(session)
        _, declared = _comfy_family(
            session,
            name="Style-routed pin",
            prompt_value="${house_style}",
            input_schema=_DECLARES_ITS_OWN,
        )
        _, undeclared = _comfy_family(session, name="Undeclared pin", prompt_value="${house_style}")
        session.commit()

    project = (await client.post("/api/projects", json={"name": "Pinned style"})).json()

    async def turn_after_pinning(revision_id: str, title: str) -> str:
        pinned = await client.put(
            f"/api/projects/{project['id']}/workflow-selections/image",
            json={"mode": "revision", "workflow_revision_id": revision_id},
        )
        assert pinned.status_code == 200, pinned.text
        chat = (
            await client.post("/api/chats", json={"title": title, "project_id": project["id"]})
        ).json()
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "a harbour at first light", "mode": "image"},
        )
        return response.text

    assert "never reads what you type" not in await turn_after_pinning(declared, "Declared")
    # The control: the same graph without the declaration, pinned the same way.
    assert "never reads what you type" in await turn_after_pinning(undeclared, "Undeclared")


def test_the_schema_is_read_only_when_the_answer_depends_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading a stored schema is the expensive part, and most graphs skip it.

    This question is asked of every workflow while the list is built, so the
    comment beside the rule claims the schema is read only when the names could
    change the answer. That is a claim about cost, and a claim about cost that
    nothing measures drifts. These four cases are the whole space: a graph that
    binds the description, one that offers no description input at all, one
    holding plain words, and the only one where a name decides it.
    """
    reads: list[str] = []
    real = prompt_binding.substitutable_keys

    def counted(operation: str, input_schema: object) -> frozenset[str] | None:
        reads.append(operation)
        return real(operation, input_schema)

    monkeypatch.setattr(prompt_binding, "substitutable_keys", counted)

    for graph in (
        _graph("${prompt}"),
        {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "x"}}},
        _graph("a post-rain dusk metropolis"),
        _graph("   "),
    ):
        ignores_the_description("comfyui", "text_to_image", graph, None)
    assert reads == []

    ignores_the_description("comfyui", "text_to_image", _graph("${house_style}"), None)
    assert reads == ["text_to_image"]


@pytest.mark.parametrize(
    ("operation", "mine", "theirs"),
    [("text_to_video", "motion_strength", "denoise"), ("text_to_image", "denoise", "codec")],
)
def test_each_role_is_judged_against_its_own_settings(
    operation: str, mine: str, theirs: str
) -> None:
    """The question has to be asked for the right kind of work.

    Image and video workflows offer different controls, so a name that binds
    something real for one is words of the author's own for the other. Asking
    with the wrong role would refuse a working video workflow and let an image
    one through, and both mistakes look identical from inside the helper.
    """
    assert ignores_the_description("comfyui", operation, _graph("${" + mine + "}"), None) is False
    assert ignores_the_description("comfyui", operation, _graph("${" + theirs + "}"), None) is True


async def test_a_workflow_with_an_unreadable_schema_still_appears_in_the_catalog(
    client: AsyncClient, settings: Settings
) -> None:
    """The listing has to survive a row this module cannot read.

    Creating a revision does not put its settings schema through the reader this
    rule uses, so a schema that cannot be read can already be stored. Asking the
    question while the list is built is new, and it must not be able to turn one
    bad row into an empty catalog.
    """
    settings.media_engine = "comfyui"
    unreadable = {
        "type": "object",
        "properties": {"frames": {"type": "integer"}, "fps": {"type": "number"}},
        "x-lm-atelier-video-length": {"version": 1, "frames_parameter": "frames"},
    }
    with SessionLocal() as session:
        _comfy_image_profile(session)
        bad, _ = _comfy_family(
            session,
            name="Unreadable settings",
            prompt_value="${house_style}",
            input_schema=unreadable,
        )
        good, _ = _comfy_family(session, name="Ordinary", prompt_value="${prompt}")
        session.commit()

    response = await client.get("/api/workflow-families?selector_capability=image")

    assert response.status_code == 200, response.text
    cards = {item["id"]: item for item in response.json()}
    assert good in cards
    # The unreadable row is listed too, judged by the shape of its value the way
    # the tool judged every workflow before names could be asked for.
    assert cards[bad]["variants"][0]["readiness_reason"] != "revision_ignores_the_description"


def test_a_setting_the_panel_hides_is_still_a_binding() -> None:
    """What a person may not change is still what the run sends.

    A workflow marks a setting read-only when the caller has no business
    choosing it - and every workflow this tool compiles for itself does exactly
    that to the runtime names it did not bind. The run supplies the value all
    the same, so a graph binding one of those names is binding something real,
    and the rule must not learn about names from the panel.
    """
    hidden = {
        "type": "object",
        "properties": {
            "seed": {"type": "integer", "readOnly": True},
            "house_style": {"type": "string", "default": "ink", "readOnly": True},
        },
    }
    keys = substitutable_keys("text_to_image", hidden)

    assert keys is not None
    assert {"seed", "house_style"} <= keys
    assert frozen_prompt(_graph("${seed}"), keys) is False
    assert frozen_prompt(_graph("${house_style}"), keys) is False
    assert (
        ignores_the_description("comfyui", "text_to_image", _graph("${house_style}"), hidden)
        is False
    )


def test_a_value_must_both_open_and_close_to_be_a_placeholder() -> None:
    """Half a placeholder is not one, and the half that is missing decides it.

    The compiler replaces a value only when the whole of it sits between an
    opening brace and a closing one. `${promptX` names nothing - the braces
    travel to the model and the description is lost - but the text after the
    opener happens to be a real name, so a rule that stopped checking the end of
    the value would read it as a binding and let the workflow through. That is
    the same defect this rule exists to close, one character away.
    """
    keys = substitutable_keys("text_to_image", None)

    assert frozen_prompt(_graph("${promptX"), keys) is True
    assert frozen_prompt(_graph("prompt}"), keys) is True
    assert frozen_prompt(_graph("$prompt}"), keys) is True
    # And the opener carries its own weight. A value borrowed from some other
    # templating syntax closes the right way and opens the wrong one, while the
    # text two characters in happens to spell a name the run really supplies -
    # so a rule that stopped checking the opening would read this as a binding
    # and let the workflow through.
    assert frozen_prompt(_graph("{{prompt}"), keys) is True
    assert frozen_prompt(_graph("${prompt}"), keys) is False


def test_a_schema_that_is_empty_rather_than_absent_is_still_unreadable() -> None:
    """An empty list is not a workflow that declares nothing.

    A stored schema of `[]` or `""` is the shape of a row written wrong, not the
    shape of a workflow with no settings. Reading it as the latter would hand
    back the full set of built-in names on the strength of a value nobody
    intended, so it is treated as unreadable and the shape test stands in.
    """
    for empty in ([], "", 0, False, 0.0):
        assert substitutable_keys("text_to_image", empty) is None, empty
    # And a schema that really does declare nothing is a different answer.
    assert substitutable_keys("text_to_image", {"type": "object", "properties": {}}) is not None
