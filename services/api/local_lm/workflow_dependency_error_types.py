"""Codes emitted while inspecting, binding, and offering workflow dependencies."""

from typing import Literal

WorkflowDependencyErrorCode = Literal[
    "dependency_data_too_deep",
    "dependency_data_too_large",
    "duplicate_dependency_requirement",
    "duplicate_dependency_slot",
    "invalid_portable_dependency_data",
    "invalid_workflow_dependencies",
    "nonportable_dependency_data",
    "too_many_workflow_dependencies",
]

WorkflowAssetBindingErrorCode = Literal[
    "ambiguous_plan_artifact",
    "artifact_folder_mismatch",
    "artifact_kind_mismatch",
    "artifact_not_downloadable",
    "artifact_not_found",
    "artifact_path_mismatch",
    "asset_reference_case_mismatch",
    "duplicate_artifact_binding",
    "duplicate_asset_reference",
    "duplicate_asset_selection",
    "install_plan_identity_mismatch",
    "install_plan_not_found",
    "install_plan_not_pending",
    "install_plan_not_supported",
    "invalid_artifact_path",
    "invalid_asset_reference",
    "invalid_asset_selection",
    "invalid_install_plan",
    "missing_asset_selection",
    "mutable_install_plan",
    "too_many_asset_bindings",
    "unexpected_asset_selection",
    "unsupported_asset_reference",
    "unverified_plan_artifact",
]

WorkflowAssetDownloadErrorCode = Literal[
    "ambiguous_install_artifact",
    "binding_asset_changed",
    "binding_plan_changed",
    "duplicate_asset_download",
    "incomplete_artifact_source",
    "incomplete_civitai_provenance",
    "install_contract_changed",
    "install_plan_changed",
    "install_plan_not_found",
    "install_plan_not_pending",
    "install_plan_not_supported",
    "invalid_install_artifact",
    "invalid_install_plan",
    "too_many_asset_downloads",
    "unsupported_install_provider",
    "unverified_install_artifact",
    "workflow_contract_changed",
]

WorkflowAssetAliasErrorCode = (
    Literal[
        "duplicate_alias_source",
        "invalid_artifact_path",
        "invalid_asset_reference",
        "invalid_install_plan",
        "nested_asset_alias",
        "unsupported_asset_runtime",
        "unverified_plan_artifact",
    ]
    | WorkflowAssetBindingErrorCode
    | WorkflowAssetDownloadErrorCode
)
