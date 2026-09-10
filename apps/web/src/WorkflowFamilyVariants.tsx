import { useState } from "react";
import type { WorkflowFamily, WorkflowInstallOffer } from "./types";
import "./WorkflowFamilyVariants.css";
import { availableWorkflowInstallOffer, workflowVariantReadinessLabel } from "./workflowVariantSetup";

const operationLabels: Record<string, string> = {
  text: "Text", text_to_image: "Text to image", image_to_image: "Image to image",
  text_to_video: "Text to video", image_to_video: "Image to video", video_to_video: "Video to video",
};
const reasons: Record<string, string> = {
  engine_mismatch: "This variant uses a different engine from the configured one.",
  revision_not_executable: "This revision has no executable workflow graph.",
  revision_untrusted: "This revision needs review before it can run.",
  activation_not_ready: "Its required dependencies are not activated for this revision.",
  model_unavailable: "Its model is not available for the configured engine.",
  operation_unavailable: "No compatible workflow revision is available for this operation.",
  current_revision_missing: "Choose or create a current revision before running.",
  family_archived: "This workflow family is archived.",
  family_disabled: "This workflow family is disabled.",
};

export function WorkflowFamilyVariants({ family, onReviewInstall }: {
  family: WorkflowFamily; onReviewInstall?: (offer: WorkflowInstallOffer, workflowName: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const variants = [...family.variants].sort((a, b) =>
    a.operation.localeCompare(b.operation) || a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  return (
    <section className="workflow-family-variants" aria-label="Operation variants">
      <h3>Operation variants</h3>
      <button className="secondary compact-button" aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}>
        {expanded ? "Hide operation variants" : "Show operation variants"}
      </button>
      {expanded && (variants.length === 0 ? <p>No operation variants.</p> : (
        <ul>{variants.map((variant) => (
          <li key={variant.id}>
            <strong>{variant.name}</strong>
            <span>{operationLabels[variant.operation] ?? variant.operation}</span>
            <small>{variant.current_revision_version === null
              ? "No current revision" : `Current revision: v${variant.current_revision_version}`}</small>
            <span className="badge">{workflowVariantReadinessLabel(variant)}</span>
            <p>{variant.readiness === "ready" ? "Ready to run."
              : Object.hasOwn(reasons, variant.readiness_reason ?? "")
                ? reasons[variant.readiness_reason ?? ""] : "No further readiness details are available."}</p>
            {onReviewInstall && availableWorkflowInstallOffer(family, variant) && (
              <button className="secondary compact-button" aria-label={`Review downloads for ${variant.name}`}
                onClick={() => {
                  const offer = availableWorkflowInstallOffer(family, variant);
                  if (offer) onReviewInstall(offer, variant.name);
                }}>Review downloads</button>
            )}
          </li>
        ))}</ul>
      ))}
    </section>
  );
}
