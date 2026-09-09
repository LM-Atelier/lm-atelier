"""Codes shared by package analysis and its compilation error subclass."""

from typing import Literal

from .comfy_package_widgets import PackageWidgetErrorCode
from .workflow_dependency_error_types import SubgraphExpansionErrorCode

# Compilation errors inherit the package exception and its constructor.
WorkflowGraphErrorCode = (
    Literal[
        "ambiguous_input_order",
        "ambiguous_widget",
        "dangling_link",
        "duplicate_input_link",
        "duplicate_input_slot",
        "duplicate_link",
        "duplicate_named_wire",
        "duplicate_node",
        "duplicate_subgraph",
        "empty_executable_workflow",
        "empty_workflow",
        "frontend_node_link",
        "inconsistent_link",
        "invalid_frontend_version",
        "invalid_identifier",
        "invalid_input_slot",
        "invalid_json",
        "invalid_key",
        "invalid_link",
        "invalid_link_slot",
        "invalid_named_wire",
        "invalid_node",
        "invalid_node_definition",
        "invalid_node_mode",
        "invalid_node_properties",
        "invalid_node_type",
        "invalid_object_info",
        "invalid_string",
        "invalid_structure",
        "invalid_subgraph",
        "invalid_subgraphs",
        "invalid_value",
        "invalid_widget_choice",
        "invalid_widget_values",
        "missing_node_type",
        "missing_required_input",
        "non_finite_number",
        "package_serialized_widgets",
        "pass_through_cycle",
        "too_deep",
        "too_large",
        "too_many_links",
        "too_many_nodes",
        "too_many_subgraphs",
        "too_many_values",
        "unconnected_pass_through",
        "undefined_named_wire",
        "unknown_input_slot",
        "unknown_widget_value",
        "unsupported_format",
        "unsupported_frontend_node",
        "unsupported_frontend_version",
        "unsupported_node_mode",
        "unsupported_primitive_node",
        "unsupported_subgraphs",
        "unsupported_widget",
        "unsupported_widget_values",
        "workflow_cycle",
    ]
    | SubgraphExpansionErrorCode
    | PackageWidgetErrorCode
)
