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
};

export function preparationErrorDescription(code: string): string {
  return PREPARATION_ERROR_DESCRIPTIONS[code] ?? code.replaceAll("_", " ");
}
