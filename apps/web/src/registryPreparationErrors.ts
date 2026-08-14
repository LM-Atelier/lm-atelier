/** Plain words for the preparation lifecycle's stable refusal codes. */
export const PREPARATION_ERROR_DESCRIPTIONS: Record<string, string> = {
  media_worker_running: "Stop the media worker before preparing packages",
  invalid_resolution: "The package's Registry record could not be verified",
  invalid_dependency_closure: "The dependency set failed verification",
  dependency_closure_incomplete: "The dependency set is not fully resolved yet",
  dependency_closure_mismatch: "The dependency set does not match this package",
  invalid_managed_root: "The managed install folder is not usable",
  managed_root_unavailable: "The managed install folder could not be reached",
  registry_install_exists: "This package is already installed",
  node_destination_exists: "The node folder already exists on disk",
  wheel_stage_exists: "A previous staging attempt was left behind",
  unbound_wheel_environment: "The wheel environment is not bound to a record",
  wheel_stage_identity_mismatch: "Staged wheels do not match their manifest",
  binding_incomplete: "The install record could not be completed",
  cleanup_failed: "Preparation failed and cleanup also failed; check the logs",
  registry_install_active: "Deactivate the package before refreshing its dependencies",
  registry_install_identity_changed: "The package changed upstream; remove and review it again",
  registry_install_review_missing: "Review the package again before refreshing its dependencies",
  renewal_cleanup_pending: "A previous dependency refresh still needs cleanup",
};

export function preparationErrorDescription(code: string): string {
  return PREPARATION_ERROR_DESCRIPTIONS[code] ?? code.replaceAll("_", " ");
}

/** Plain words for the trust and activation refusal codes. */
export const ACTIVATION_ERROR_DESCRIPTIONS: Record<string, string> = {
  media_worker_running: "Stop the media worker before changing package trust or activation",
  registry_install_not_found: "This prepared package no longer exists",
  registry_install_untrusted: "Trust the package before activating it",
  registry_install_verification_failed: "The package's files or dependencies failed verification",
  activation_start_failed: "The package broke media startup; the prior runtime was restored",
  activation_restore_failed: "Activation failed and the prior runtime did not restart; check the logs",
  activation_cancelled: "Activation was cancelled before the runtime came up",
  deactivation_restart_failed: "The package is inactive but the runtime did not restart; check the logs",
  "registry-install-files-missing": "Remove this incomplete package and prepare it again",
  registry_install_active: "Deactivate the package before removing it",
  registry_install_in_use: "A workflow still depends on this package",
  registry_install_busy: "Wait for the dependency refresh to finish before removing it",
  registry_install_path_invalid: "The managed package folder is not safe to remove",
  registry_install_remove_failed: "The package could not be removed; check the logs",
  registry_install_restore_failed: "Removal failed and the package files could not be restored",
};

export function activationErrorDescription(error: { code?: string; message: string }): string {
  return (error.code && ACTIVATION_ERROR_DESCRIPTIONS[error.code]) || error.message;
}
