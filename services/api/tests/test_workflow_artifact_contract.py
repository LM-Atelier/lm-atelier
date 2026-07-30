from __future__ import annotations

from local_lm.model_planner import workflow_artifact_contract

_BASE = {
    "operation": "text_to_image",
    "engine": "comfyui",
    "api_graph": {"3": {"class_type": "KSampler", "inputs": {"steps": 20, "cfg": 7.0}}},
    "input_schema": {"type": "object", "properties": {"steps": {"type": "integer"}}},
    "dependencies": {
        "template_id": "example_template",
        "template_sha256": "a" * 64,
        "model_files": ["model.safetensors"],
        "custom_nodes": [],
        "extensions": {},
    },
}


def test_the_same_compile_produces_the_same_contract() -> None:
    """Instability here would invalidate evidence unpredictably, which is worse
    than the predictable invalidation this replaces."""
    assert workflow_artifact_contract(**_BASE) == workflow_artifact_contract(**_BASE)


def test_mapping_order_does_not_change_the_contract() -> None:
    reordered = dict(_BASE)
    reordered["api_graph"] = {"3": {"inputs": {"cfg": 7.0, "steps": 20}, "class_type": "KSampler"}}
    reordered["dependencies"] = {
        "extensions": {},
        "custom_nodes": [],
        "model_files": ["model.safetensors"],
        "template_sha256": "a" * 64,
        "template_id": "example_template",
    }

    assert workflow_artifact_contract(**reordered) == workflow_artifact_contract(**_BASE)


def test_a_changed_graph_changes_the_contract() -> None:
    changed = dict(_BASE)
    changed["api_graph"] = {"3": {"class_type": "KSampler", "inputs": {"steps": 8, "cfg": 7.0}}}

    assert workflow_artifact_contract(**changed) != workflow_artifact_contract(**_BASE)


def test_a_changed_schema_changes_the_contract() -> None:
    changed = dict(_BASE)
    changed["input_schema"] = {
        "type": "object",
        "properties": {"steps": {"type": "integer", "readOnly": True}},
    }

    assert workflow_artifact_contract(**changed) != workflow_artifact_contract(**_BASE)


def test_a_changed_execution_dependency_changes_the_contract() -> None:
    changed = dict(_BASE)
    changed["dependencies"] = {
        **_BASE["dependencies"],
        "custom_nodes": [{"id": "pack", "revision": "b" * 40}],
    }

    assert workflow_artifact_contract(**changed) != workflow_artifact_contract(**_BASE)


def test_local_identifiers_do_not_change_the_contract() -> None:
    """Install ids and the compiler version differ per machine and per release;
    including them is exactly the coupling this removes."""
    with_local = dict(_BASE)
    with_local["dependencies"] = {
        **_BASE["dependencies"],
        "model_install_ids": ["model_abc123"],
        "compiler_version": 999,
    }

    assert workflow_artifact_contract(**with_local) == workflow_artifact_contract(**_BASE)


def test_list_order_is_significant() -> None:
    """LoRA stacks and multi-file bundles apply in order, so order is semantic."""
    reordered = dict(_BASE)
    reordered["dependencies"] = {
        **_BASE["dependencies"],
        "model_files": ["second.safetensors", "model.safetensors"],
    }

    assert workflow_artifact_contract(**reordered) != workflow_artifact_contract(**_BASE)
