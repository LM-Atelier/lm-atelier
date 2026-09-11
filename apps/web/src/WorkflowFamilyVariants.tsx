import { useState } from "react";
import type { WorkflowFamily, WorkflowInstallOffer } from "./types";
import "./WorkflowFamilyVariants.css";
import { availableWorkflowInstallOffer, workflowVariantReadinessLabel } from "./workflowVariantSetup";
import { readinessReason } from "./readinessReason";

const operationLabels: Record<string, string> = {
  text: "Text", text_to_image: "Text to image", image_to_image: "Image to image",
  text_to_video: "Text to video", image_to_video: "Image to video", video_to_video: "Video to video",
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
            <p>{variant.readiness === "ready" ? "Ready to run." : readinessReason(variant)}</p>
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
