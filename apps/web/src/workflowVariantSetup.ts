import type { WorkflowFamily, WorkflowFamilyVariant, WorkflowInstallOffer } from "./types";

export function availableWorkflowInstallOffer(
  family: WorkflowFamily, variant: WorkflowFamilyVariant,
): WorkflowInstallOffer | null {
  const offer = variant.install_offer;
  return family.enabled && !family.archived && variant.readiness === "setup_required"
    && variant.setup_resolution === "reviewed_download_available"
    && offer?.status === "ready" && offer.workflow_revision_id === variant.current_revision_id
    ? offer : null;
}

export function workflowVariantReadinessLabel(variant: WorkflowFamilyVariant): string {
  if (variant.readiness === "setup_required") {
    if (variant.setup_resolution === "reviewed_download_available") return "Needs download";
    if (variant.setup_resolution === "attention_required") return "Needs attention";
    return "Needs setup";
  }
  return { ready: "Ready", review_required: "Needs review", unavailable: "Unavailable" }[variant.readiness];
}
