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
