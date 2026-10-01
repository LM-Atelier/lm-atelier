import { workflowOperationLabels } from "./workflowFamilies";
import "./WorkflowFamilyList.css";
import { WorkflowInstallStatus } from "./WorkflowInstallStatus";
import { WorkflowReadPageControls } from "./WorkflowReadPageControls";
import { useWorkflowFamilyVariantPages } from "./useWorkflowLibraryReads";
import { availableWorkflowInstallOffer, workflowVariantReadinessLabel } from "./workflowVariantSetup";
import type { WorkflowFamily, WorkflowInstallOffer, WorkflowSummary, WorkflowVariantReadiness } from "./types";

export type WorkflowLibraryChoice = Pick<WorkflowSummary, "id" | "current_revision_id">;


export function WorkflowFamilyCard({ family, selectedId, onSelect, operation, readiness,
  installationInDetails, onReviewInstall }: {
  family: WorkflowFamily; selectedId: string | null; onSelect: (workflow: WorkflowLibraryChoice) => void;
  operation?: string; readiness?: WorkflowVariantReadiness; installationInDetails?: string | null;
  onReviewInstall?: (offer: WorkflowInstallOffer, workflowName: string) => void;
}) {
  const variants = useWorkflowFamilyVariantPages(family.id, operation, readiness, family);
  return <section aria-labelledby={`workflow-family-${family.id}`}>
    <h3 id={`workflow-family-${family.id}`}>{family.name}</h3>
    {(family.use_case || family.description) && <p className="muted">{family.use_case || family.description}</p>}
    {family.use_case_derived && <p className="muted">Derived from model metadata</p>}
    {family.dependency_summary && <p className="muted">
      {family.dependency_summary.dependency_count} recorded {family.dependency_summary.dependency_count === 1 ? "dependency" : "dependencies"}
    </p>}
    {family.archived && <span className="badge">Archived</span>}
    <div className="workflow-list">
      {(variants.data ?? []).map(variant => {
        const offer = availableWorkflowInstallOffer(family, variant);
        return <div key={variant.id} className="workflow-family-variant">
          <button className={selectedId === variant.id ? "selected" : ""}
            aria-pressed={selectedId === variant.id} onClick={() => onSelect(variant)}>
            <span><strong>{variant.name}</strong>
              <small>{workflowOperationLabels[variant.operation] ?? variant.operation}
                {variant.current_revision_version !== null && ` · v${variant.current_revision_version}`}</small>
              <span className="badge">{workflowVariantReadinessLabel(variant)}</span>
            </span>
          </button>
          <WorkflowInstallStatus progress={variant.install_progress} workflowName={variant.name}
            revisionId={variant.current_revision_id} summary={variant.id === installationInDetails}
            onReviewSetup={() => onSelect(variant)} />
          {offer && onReviewInstall && <button className="workflow-family-install-action"
            aria-label={`Review downloads for ${variant.name}`}
            onClick={() => onReviewInstall(offer, variant.name)}>Review downloads</button>}
        </div>;
      })}
    </div>
    <WorkflowReadPageControls pages={variants} label={`${family.name} variants`} />
  </section>;
}
