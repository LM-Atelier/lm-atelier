from __future__ import annotations

from typing import Any

import pytest

from local_lm.comfy_workflow_packages import (
    FRONTEND_SYSTEM_NODE_TYPES,
    MAX_UI_GRAPH_DEPTH,
    MAX_UI_GRAPH_NODES,
    WorkflowPackageError,
    WorkflowPackageIssue,
    analyze_comfyui_workflow_package,
)


def node(
    identifier: int | str,
    node_type: str,
    *,
    package: str | None = None,
    version: str | None = None,
    widgets: object = (),
) -> dict[str, Any]:
    properties = {}
    if package is not None:
        properties["cnr_id"] = package
    if version is not None:
        properties["ver"] = version
    return {
        "id": identifier,
        "type": node_type,
        "properties": properties,
        "widgets_values": widgets,
    }


def workflow(
    *,
    nodes: list[dict[str, Any]] | None = None,
    links: list[object] | None = None,
    subgraphs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": nodes
        if nodes is not None
        else [node(1, "KSampler", package="comfy-core", version="0.28.0")],
        "links": links or [],
        "definitions": {"subgraphs": subgraphs or []},
        "extra": {"frontendVersion": "1.45.21"},
    }


def test_analyzer_separates_frontend_subgraph_and_runtime_nodes() -> None:
    subgraph_id = "9bc44576-7290-4701-bda4-032ca796efbc"
    value = workflow(
        nodes=[
            node(1, "PrimitiveNode"),
            node(2, subgraph_id),
            node(3, "KSampler", package="comfy-core", version="0.28.0"),
            node(4, "Power Lora Loader", package="rgthree-comfy", version="1.2.3"),
        ],
        links=[[1, 1, 0, 3, 0, "*"]],
        subgraphs=[
            {
                "id": subgraph_id,
                "nodes": [node(10, "CLIPTextEncode", package="comfy-core", version="0.28.0")],
                "links": [
                    {
                        "id": 2,
                        "origin_id": -10,
                        "target_id": 10,
                    },
                    {
                        "id": 3,
                        "origin_id": 10,
                        "target_id": -20,
                    },
                ],
            }
        ],
    )
    analysis = analyze_comfyui_workflow_package(
        value, available_node_types={"KSampler", "CLIPTextEncode"}
    )
    assert analysis.frontend_version == "1.45.21"
    assert analysis.node_count == 5
    assert analysis.link_count == 3
    assert analysis.frontend_node_types == (subgraph_id, "PrimitiveNode")
    assert analysis.required_node_types == (
        "CLIPTextEncode",
        "KSampler",
        "Power Lora Loader",
    )
    assert analysis.missing_node_types == ("Power Lora Loader",)
    assert analysis.missing_nodes[0].node_type == "Power Lora Loader"
    assert analysis.missing_nodes[0].count == 1
    assert analysis.missing_nodes[0].package_id == "rgthree-comfy"
    assert analysis.custom_packages[0].package_id == "rgthree-comfy"
    assert analysis.custom_packages[0].versions == ("1.2.3",)
    assert not analysis.custom_packages[0].locally_resolved
    assert analysis.operation_guess == "image"
    assert not analysis.truncated
    assert not analysis.ready
    assert not analysis.dependencies_resolved


def test_all_available_dependencies_are_resolved() -> None:
    value = workflow(
        nodes=[
            node(1, "KSampler", package="comfy-core", version="0.28.0"),
            node(2, "CustomNode", package="example-pack", version="a" * 40),
        ]
    )
    analysis = analyze_comfyui_workflow_package(
        value,
        available_node_types={"KSampler", "CustomNode"},
        installed_package_versions={"example-pack": {"a" * 40}},
    )
    assert analysis.runtime_nodes_available
    assert analysis.dependencies_resolved
    assert analysis.custom_packages[0].package_id == "example-pack"
    assert analysis.custom_packages[0].versions == ("a" * 40,)
    assert analysis.custom_packages[0].locally_resolved
    assert analysis.issues == ()
    assert analysis.ready


def test_asset_inventory_is_data_only_and_never_keeps_remote_urls() -> None:
    value = workflow(
        nodes=[
            node(
                1,
                "Loader",
                package="comfy-core",
                widgets=[
                    "models/portrait.safetensors",
                    "unsafe.ckpt",
                    "weights/model.onnx",
                    "https://example.invalid/private.safetensors?token=secret",
                    "../escape.safetensors",
                    "nested/../escape.safetensors",
                    "model.safetensors?version=1",
                ],
            )
        ]
    )
    analysis = analyze_comfyui_workflow_package(
        value,
        available_node_types={"Loader"},
        available_asset_filenames={"models/portrait.safetensors"},
    )
    assert [(item.filename, item.policy) for item in analysis.asset_references] == [
        ("models/portrait.safetensors", "supported"),
        ("unsafe.ckpt", "blocked"),
        ("weights/model.onnx", "unsupported"),
    ]
    assert analysis.asset_references[0].kind == "checkpoint"
    assert analysis.asset_references[0].source_url is None
    assert analysis.asset_references[0].present_locally
    assert {issue.code: issue.count for issue in analysis.issues} == {
        "blocked_asset_format": 1,
        "remote_url_reference": 1,
        "unsafe_asset_reference": 3,
        "unsupported_asset_format": 1,
    }
    assert "secret" not in repr(analysis)
    assert not analysis.dependencies_resolved


def test_missing_node_without_package_metadata_is_reported() -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(nodes=[node(1, "UnknownNode")]),
        available_node_types=set(),
    )
    assert analysis.missing_node_types == ("UnknownNode",)
    assert analysis.issues[0].code == "unidentified_custom_node_package"
    assert analysis.issues[0].node_types == ("UnknownNode",)


def test_frontend_notes_do_not_create_executable_asset_dependencies() -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[
                node(
                    1,
                    "MarkdownNote",
                    widgets=["Documentation: https://example.invalid/model.safetensors"],
                ),
                node(2, "KSampler"),
            ]
        ),
        available_node_types={"KSampler"},
    )
    assert analysis.asset_references == ()
    assert analysis.issues == ()


def test_missing_supported_asset_blocks_readiness() -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[
                node(
                    1,
                    "LoraLoader",
                    package="comfy-core",
                    widgets=["styles/detail.safetensors"],
                )
            ]
        ),
        available_node_types={"LoraLoader"},
    )
    assert analysis.asset_references[0].kind == "lora"
    assert not analysis.asset_references[0].present_locally
    assert analysis.issues == (WorkflowPackageIssue("missing_asset", 1),)
    assert not analysis.ready


def test_unversioned_custom_package_is_reported() -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(nodes=[node(1, "CustomNode", package="example-pack")]),
        available_node_types=set(),
    )
    assert analysis.custom_packages[0].versions == ()
    assert {issue.code for issue in analysis.issues} == {
        "unresolved_custom_node_package",
        "unversioned_custom_node_package",
    }


def test_conflicting_custom_package_versions_are_reported() -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[
                node(1, "FirstCustomNode", package="example-pack", version="1.0.0"),
                node(2, "SecondCustomNode", package="example-pack", version="2.0.0"),
            ]
        ),
        available_node_types={"FirstCustomNode", "SecondCustomNode"},
    )
    assert analysis.custom_packages[0].versions == ("1.0.0", "2.0.0")
    assert {issue.code for issue in analysis.issues} == {
        "conflicting_custom_node_versions",
        "unresolved_custom_node_package",
    }
    conflict = next(
        issue for issue in analysis.issues if issue.code == "conflicting_custom_node_versions"
    )
    assert conflict == WorkflowPackageIssue(
        "conflicting_custom_node_versions",
        2,
        ("FirstCustomNode", "SecondCustomNode"),
    )
    assert not analysis.dependencies_resolved


def test_a_canvas_label_never_becomes_a_runtime_dependency() -> None:
    """rgthree draws its label in the browser and registers no node for it.

    Installing the package it names cannot satisfy it, so counting it as
    missing refuses the workflow over a caption.
    """

    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[
                node(1, "Label (rgthree)"),
                node(2, "Lora Loader Stack (rgthree)"),
            ]
        ),
        available_node_types={"Lora Loader Stack (rgthree)"},
    )

    assert analysis.runtime_nodes_available
    assert "Label (rgthree)" in analysis.frontend_node_types


def test_a_named_wire_never_makes_the_package_that_draws_it_a_dependency() -> None:
    """`SetNode` and `GetNode` are KJNodes JavaScript with no Python class.

    They carry a package id, so counting them as runtime nodes attributed them
    to that package and then reported it as the package that failed to load -
    while the package was installed, trusted, and loading correctly. Nothing can
    satisfy them, so a graph using the idiom could never be imported no matter
    what was installed.
    """

    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[
                node(1, "SetNode", package="comfyui-kjnodes", version="1.2.3"),
                node(2, "GetNode", package="comfyui-kjnodes", version="1.2.3"),
                node(3, "KSampler"),
            ]
        ),
        available_node_types={"KSampler"},
    )

    assert analysis.runtime_nodes_available
    assert analysis.missing_node_types == ()
    assert set(analysis.frontend_node_types) == {"GetNode", "SetNode"}
    assert [package.package_id for package in analysis.custom_packages] == []


@pytest.mark.parametrize("node_type", sorted(FRONTEND_SYSTEM_NODE_TYPES))
def test_frontend_system_nodes_are_not_runtime_dependencies(node_type: str) -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(nodes=[node(1, node_type), node(2, "KSampler")]),
        available_node_types={"KSampler"},
    )
    assert node_type in analysis.frontend_node_types
    assert analysis.missing_node_types == ()


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update(version=0.3), "unsupported_format"),
        (lambda value: value.update(version=[]), "unsupported_format"),
        (lambda value: value.update(nodes=[]), "empty_workflow"),
        (
            lambda value: value["nodes"].append(node(1, "Duplicate")),
            "duplicate_node",
        ),
        (
            lambda value: value.update(
                definitions={
                    "subgraphs": [
                        {"id": "same", "nodes": [], "links": []},
                        {"id": "same", "nodes": [], "links": []},
                    ]
                }
            ),
            "duplicate_subgraph",
        ),
    ],
)
def test_malformed_graphs_fail_before_dependency_resolution(
    mutation: Any,
    code: str,
) -> None:
    value = workflow()
    mutation(value)
    with pytest.raises(WorkflowPackageError) as captured:
        analyze_comfyui_workflow_package(value)
    assert captured.value.code == code


def test_dangling_ui_links_are_inert_and_reported() -> None:
    value = workflow(links=[[1, 1, 0, 999, 0, "IMAGE"]])
    analysis = analyze_comfyui_workflow_package(
        value,
        available_node_types={"KSampler"},
    )
    assert analysis.issues == (WorkflowPackageIssue("dangling_link", 1, severity="advisory"),)
    assert analysis.dependencies_resolved
    assert analysis.ready


def test_h3_nodes_are_observed_without_authority() -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[
                node(1, "MiniMaxH3ImageToVideo", package="comfy-core", version="0.30.0"),
                node(2, "KSampler", package="comfy-core", version="0.30.0"),
            ]
        ),
        available_node_types={"MiniMaxH3ImageToVideo", "KSampler"},
    )
    assert len(analysis.model_family_observations) == 1
    observation = analysis.model_family_observations[0]
    assert observation.family_id == "minimax-h3"
    assert observation.declaration_source == "exact_workflow_node_types"
    assert observation.evidence_node_types == ("MiniMaxH3ImageToVideo",)
    assert observation.variant_hints == ("image_to_video",)
    assert observation.runtime_verified is False
    assert observation.installation_authorized is False
    assert observation.execution_authorized is False
    assert observation.workflow_contract_bound is False
    assert observation.geometry_claimed is False
    assert observation.reference_slots_claimed is False
    assert not hasattr(observation, "geometry_alignment_pixels")
    assert not hasattr(observation, "reference_image_slots_max")
    assert analysis.ready


def test_h3_lookalikes_are_not_observed() -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(nodes=[node(1, "MiniMaxH3ImageToVideoX")]),
        available_node_types={"MiniMaxH3ImageToVideoX"},
    )
    assert analysis.model_family_observations == ()


def test_non_h3_graph_has_no_family_observation() -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(nodes=[node(1, "KSampler")]),
        available_node_types={"KSampler"},
    )
    assert analysis.model_family_observations == ()


def test_operation_guess_is_display_only() -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(nodes=[node(1, "VHS_VideoCombine")]),
        available_node_types={"VHS_VideoCombine"},
    )
    assert analysis.operation_guess == "video"
    assert analysis.ready


@pytest.mark.parametrize(
    ("node_types", "expected"),
    [
        (("PreviewAny",), "unknown"),
        (("PreviewAny", "KSampler"), "image"),
        (("WanVideoModelLoader",), "video"),
        (("LTXVLoader",), "video"),
        (("HunyuanVideoSampler",), "video"),
    ],
)
def test_operation_guess_matches_node_tokens(
    node_types: tuple[str, ...],
    expected: str,
) -> None:
    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[node(index, node_type) for index, node_type in enumerate(node_types, start=1)]
        ),
        available_node_types=set(node_types),
    )

    assert analysis.operation_guess == expected


def test_node_bound_is_enforced() -> None:
    value = workflow(nodes=[node(index, "KSampler") for index in range(MAX_UI_GRAPH_NODES + 1)])
    with pytest.raises(WorkflowPackageError) as captured:
        analyze_comfyui_workflow_package(value)
    assert captured.value.code == "too_many_nodes"


def test_depth_bound_is_enforced_without_recursive_parsing() -> None:
    value = workflow()
    current: list[object] = []
    value["extra"] = current
    for _ in range(MAX_UI_GRAPH_DEPTH + 1):
        nested: list[object] = []
        current.append(nested)
        current = nested
    with pytest.raises(WorkflowPackageError) as captured:
        analyze_comfyui_workflow_package(value)
    assert captured.value.code == "too_deep"


def test_non_finite_numbers_are_refused() -> None:
    value = workflow()
    value["extra"] = {"scale": float("nan")}
    with pytest.raises(WorkflowPackageError) as captured:
        analyze_comfyui_workflow_package(value)
    assert captured.value.code == "non_finite_number"


def test_rgthree_group_controls_are_frontend_even_with_their_links() -> None:
    """The live PhotoFlow blockers, in the shape the graph actually carries.

    These three set other nodes' modes while a graph is being edited and have
    no runtime existence. They omit `cnr_id`, the bypasser exposes only an
    `OPT_CONNECTION` output, and the relay and repeater form a virtual
    `REPEATER` chain between themselves - so a name-only assertion would not
    have exercised the links that made them look like real dependencies.
    """
    bypasser = node(10, "Fast Groups Bypasser (rgthree)")
    bypasser["outputs"] = [{"name": "OPT_CONNECTION", "type": "OPT_CONNECTION", "links": [1]}]
    relay = node(11, "Mute / Bypass Relay (rgthree)")
    relay["inputs"] = [{"name": "OPT_CONNECTION", "type": "OPT_CONNECTION", "link": 1}]
    relay["outputs"] = [{"name": "REPEATER", "type": "REPEATER", "links": [2]}]
    repeater = node(12, "Mute / Bypass Repeater (rgthree)")
    repeater["inputs"] = [{"name": "REPEATER", "type": "REPEATER", "link": 2}]

    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[bypasser, relay, repeater, node(1, "KSampler")],
            links=[
                [1, 10, 0, 11, 0, "OPT_CONNECTION"],
                [2, 11, 0, 12, 0, "REPEATER"],
            ],
        ),
        available_node_types={"KSampler"},
    )

    assert analysis.missing_node_types == ()
    for control in (
        "Fast Groups Bypasser (rgthree)",
        "Mute / Bypass Relay (rgthree)",
        "Mute / Bypass Repeater (rgthree)",
    ):
        assert control in analysis.frontend_node_types
    # Narrow on purpose: a node with no package is usually one whose package
    # we failed to identify, and treating that class as furniture would drop
    # real dependencies silently.
    unknown = analyze_comfyui_workflow_package(
        workflow(nodes=[node(20, "SomeOtherUnpackagedNode"), node(1, "KSampler")]),
        available_node_types={"KSampler"},
    )
    assert unknown.missing_node_types == ("SomeOtherUnpackagedNode",)


def test_a_node_with_no_attribution_is_not_the_same_claim_as_core() -> None:
    """Everything reading provenance today collapses these two. The published
    Krea 2 workflow names `Save Image (LoraManager)` with no cnr_id at all, and
    treating that as core would admit it as though the runtime shipped it."""

    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[
                node(1, "KSampler", package="comfy-core", version="0.28.0"),
                node(2, "Save Image (LoraManager)"),
                node(3, "Lora Loader Stack (rgthree)", package="rgthree-comfy", version="1.2.3"),
            ]
        ),
        available_node_types={
            "KSampler",
            "Save Image (LoraManager)",
            "Lora Loader Stack (rgthree)",
        },
    )
    provenance = {item.node_type: item for item in analysis.node_provenance}

    assert provenance["KSampler"].core_claimed
    assert not provenance["KSampler"].unattributed

    assert provenance["Save Image (LoraManager)"].unattributed
    assert not provenance["Save Image (LoraManager)"].core_claimed
    assert provenance["Save Image (LoraManager)"].package_ids == ()

    assert provenance["Lora Loader Stack (rgthree)"].package_ids == ("rgthree-comfy",)
    assert provenance["Lora Loader Stack (rgthree)"].package_versions == ("1.2.3",)
    assert not provenance["Lora Loader Stack (rgthree)"].unattributed


def test_provenance_reports_and_decides_nothing() -> None:
    """It must not feed readiness: import has to behave exactly as before."""

    unattributed = workflow(nodes=[node(1, "KSampler")])
    attributed = workflow(nodes=[node(1, "KSampler", package="comfy-core", version="0.28.0")])

    first = analyze_comfyui_workflow_package(unattributed, available_node_types={"KSampler"})
    second = analyze_comfyui_workflow_package(attributed, available_node_types={"KSampler"})

    assert first.ready == second.ready
    assert first.runtime_nodes_available == second.runtime_nodes_available
    assert first.missing_node_types == second.missing_node_types
    # ...while still telling them apart.
    assert first.node_provenance[0].unattributed
    assert second.node_provenance[0].core_claimed


def test_a_package_trusted_before_its_inventory_was_recorded_says_so() -> None:
    """Absent evidence resolves exactly as poorly as an absent package and needs
    the opposite action: one is fixed by fetching something, the other by
    reading what is already installed.

    This is reachable by upgrading rather than by doing anything wrong. The
    reviewed inventory began being recorded after people had already trusted
    packages, so every such install reports as unresolved until somebody reads
    that exact revision again - and until this distinction existed, the
    application sent them to install what they already had.
    """

    value = workflow(
        nodes=[
            node(1, "KSampler", package="comfy-core", version="0.28.0"),
            node(2, "CustomNode", package="example-pack", version="a" * 40),
        ]
    )

    analysis = analyze_comfyui_workflow_package(
        value,
        available_node_types={"KSampler", "CustomNode"},
        packages_awaiting_review={("example-pack", "a" * 40)},
    )

    codes = {issue.code for issue in analysis.issues}
    assert "custom_node_package_awaiting_review" in codes
    assert "unresolved_custom_node_package" not in codes
    # Still blocking: the workflow genuinely cannot run. Only the remedy differs.
    assert not analysis.dependencies_resolved
    assert not analysis.ready


def test_an_absent_package_is_still_reported_as_absent() -> None:
    """The new state must not swallow the old one, or a package nobody
    installed would be reported as merely needing a read."""

    value = workflow(
        nodes=[
            node(1, "KSampler", package="comfy-core", version="0.28.0"),
            node(2, "CustomNode", package="example-pack", version="a" * 40),
        ]
    )

    analysis = analyze_comfyui_workflow_package(
        value,
        available_node_types={"KSampler", "CustomNode"},
        packages_awaiting_review=set(),
    )

    codes = {issue.code for issue in analysis.issues}
    assert "unresolved_custom_node_package" in codes
    assert "custom_node_package_awaiting_review" not in codes


def test_a_reviewed_package_is_neither() -> None:
    """Recording the inventory is what makes the package resolve, so a package
    that has it must report no issue at all."""

    value = workflow(
        nodes=[
            node(1, "KSampler", package="comfy-core", version="0.28.0"),
            node(2, "CustomNode", package="example-pack", version="a" * 40),
        ]
    )

    analysis = analyze_comfyui_workflow_package(
        value,
        available_node_types={"KSampler", "CustomNode"},
        installed_package_versions={"example-pack": {"a" * 40}},
        packages_awaiting_review={("example-pack", "a" * 40)},
    )

    assert analysis.issues == ()
    assert analysis.ready


_RGTHREE_AUDITED_REVISION = "1.0.2605082257"


def _lora(name: str, *, on: bool = True, strength: float = 1, strength_two: object = ...) -> Any:
    entry: dict[str, Any] = {"on": on, "lora": name, "strength": strength}
    if strength_two is not ...:
        entry["strengthTwo"] = strength_two
    return entry


def _loader(*entries: Any) -> dict[str, Any]:
    return node(
        1,
        "Power Lora Loader (rgthree)",
        package="rgthree-comfy",
        version=_RGTHREE_AUDITED_REVISION,
        widgets=[{}, {"type": "PowerLoraLoaderHeaderWidget"}, *entries, {}, ""],
    )


def _analysed(*entries: Any, available: tuple[str, ...] = ()) -> Any:
    return analyze_comfyui_workflow_package(
        workflow(nodes=[_loader(*entries), node(2, "KSampler")]),
        available_node_types={"KSampler", "Power Lora Loader (rgthree)"},
        available_asset_filenames=available,
    )


def test_a_lora_the_node_would_load_is_a_dependency() -> None:
    analysis = _analysed(_lora("wanted.safetensors"))

    assert [asset.filename for asset in analysis.asset_references] == ["wanted.safetensors"]
    assert any(issue.code == "missing_asset" for issue in analysis.issues)


@pytest.mark.parametrize(
    "entry",
    [
        _lora("held.safetensors", on=False),
        _lora("held.safetensors", strength=0),
        _lora("held.safetensors", strength=0, strength_two=0),
    ],
)
def test_a_lora_the_node_would_not_load_is_not_a_dependency(entry: Any) -> None:
    """An entry switched off, or left at zero strength, keeps its filename so the
    author can turn it back on. The node never opens it, so counting it blocked
    the whole import over an asset no run would read."""

    analysis = _analysed(entry)

    assert analysis.asset_references == ()
    assert not any(issue.code == "missing_asset" for issue in analysis.issues)


def test_a_zero_model_strength_with_a_clip_strength_is_still_a_dependency() -> None:
    """The node applies either one, so only both being zero means it loads
    nothing."""

    analysis = _analysed(_lora("wanted.safetensors", strength=0, strength_two=0.4))

    assert [asset.filename for asset in analysis.asset_references] == ["wanted.safetensors"]


def test_a_mixed_loader_reports_only_what_it_would_load() -> None:
    analysis = _analysed(
        _lora("live.safetensors"),
        _lora("held.safetensors", on=False),
        _lora("muted.safetensors", strength=0),
    )

    assert [asset.filename for asset in analysis.asset_references] == ["live.safetensors"]


def test_a_loader_this_build_cannot_read_still_reports_every_filename() -> None:
    """Over-reporting a dependency is a worse answer than under-reporting one,
    and a revision nobody audited is not evidence that an entry is dormant."""

    unread = node(
        1,
        "Power Lora Loader (rgthree)",
        package="rgthree-comfy",
        version="1.0.9999999999",
        widgets=[
            {},
            {"type": "PowerLoraLoaderHeaderWidget"},
            _lora("held.safetensors", on=False),
            {},
            "",
        ],
    )
    analysis = analyze_comfyui_workflow_package(
        workflow(nodes=[unread, node(2, "KSampler")]),
        available_node_types={"KSampler", "Power Lora Loader (rgthree)"},
    )

    assert [asset.filename for asset in analysis.asset_references] == ["held.safetensors"]


@pytest.mark.parametrize(
    "node_type",
    [
        "Fast Groups Muter (rgthree)",
        "Fast Actions Button (rgthree)",
        "Bookmark (rgthree)",
    ],
)
def test_an_editor_control_is_not_a_node_the_runtime_must_provide(node_type: str) -> None:
    """These set other nodes' modes, or move the view, while a graph is being
    edited. The runtime registers none of them, so counting one as missing sends
    someone to install a package that is already installed and could not have
    helped - the muter especially, which sits beside a bypasser that was already
    named here.
    """

    analysis = analyze_comfyui_workflow_package(
        workflow(
            nodes=[
                node(1, node_type, package="rgthree-comfy", version="1.0.2605082257"),
                node(2, "KSampler"),
            ]
        ),
        available_node_types={"KSampler"},
    )

    assert analysis.missing_node_types == ()
    assert node_type in analysis.frontend_node_types


def test_a_collector_carries_a_wire_and_stays_a_runtime_dependency() -> None:
    """It registers nothing either, but it gathers several connections into one
    output. Calling it furniture would drop the edges it stands for."""

    assert "Node Collector (rgthree)" not in FRONTEND_SYSTEM_NODE_TYPES
