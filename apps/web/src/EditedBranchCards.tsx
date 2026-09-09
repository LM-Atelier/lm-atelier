import { jobProgressText } from "./jobProgress";
import type { EditedBranch, Job, WorkStep } from "./types";

export interface EditedBranchCardsProps {
  branches: EditedBranch[];
  activeHeadId: string | null;
  activatingPlanId?: string | null;
  onView: (branch: EditedBranch) => void;
  onContinue: (branch: EditedBranch) => void;
  onCancelPlan: (planId: string) => void;
  onRetryPlan: (planId: string) => void;
  onCancelStep: (stepId: string) => void;
  onRetryStep: (stepId: string) => void;
}

const ACTIVE = new Set(["queued", "running", "paused"]);
const UNSUCCESSFUL = new Set(["failed", "cancelled", "interrupted"]);

function matchesStep(job: Job, step: WorkStep): boolean {
  return job.work_step_id ? job.work_step_id === step.id : Boolean(step.run_id && job.run_id === step.run_id);
}

function progressText(job: Job): string {
  const resource = job.progress_json?.queue_resource ?? job.queue_resource;
  const label = resource ? resource.replaceAll("_", " ") : "";
  return [label, jobProgressText(job)].filter(Boolean).join(" · ");
}

/** Alternate versions remain discoverable without changing the current conversation. */
export function EditedBranchCards({ branches, activeHeadId, activatingPlanId, onView, onContinue, onCancelPlan, onRetryPlan, onCancelStep, onRetryStep }: EditedBranchCardsProps) {
  if (!branches.length) return null;
  return (
    <section aria-label="Edited versions">
      {branches.map((branch, index) => {
        const { plan } = branch;
        const steps = [...plan.steps].sort((left, right) => left.ordinal - right.ordinal);
        const jobs = branch.jobs.filter((job) => job.work_plan_id === plan.id && steps.some((step) => matchesStep(job, step)));
        const activeJobs = jobs.filter((job) => ACTIVE.has(job.status));
        // The plan endpoint acts on every active job; mixed availability belongs to step controls.
        const canCancelPlan = activeJobs.length > 0 && activeJobs.every((job) => job.cancellable);
        const canRetryPlan = jobs.some((job) => UNSUCCESSFUL.has(job.status));
        const active = activeHeadId === branch.branch_head_message_id;
        return (
          <article className="media-output-plan" aria-label={`Edited version ${index + 1}`} key={plan.id}>
            <div className="message-meta" aria-live="polite">
              <strong>Edited version {plan.status.replaceAll("_", " ")}</strong>
              {active && <span>Current version</span>}
              {!branch.source_available && <span>Original message unavailable</span>}
            </div>
            <div className="media-output-plan-actions">
              {branch.branch_head_message_id && <button className="secondary compact-button" onClick={() => onView(branch)}>View edited branch</button>}
              <button className="secondary compact-button" disabled={!branch.can_continue || active || Boolean(activatingPlanId) || !branch.branch_head_message_id}
                onClick={() => onContinue(branch)}>Continue from this version</button>
              {canCancelPlan && <button className="secondary compact-button" onClick={() => onCancelPlan(plan.id)}>Cancel edited version</button>}
              {canRetryPlan && <button className="secondary compact-button" onClick={() => onRetryPlan(plan.id)}>Retry unsuccessful outputs</button>}
            </div>
            <ol>
              {steps.map((step) => {
                const stepJobs = jobs.filter((job) => matchesStep(job, step));
                const activeStepJobs = stepJobs.filter((job) => ACTIVE.has(job.status));
                const canCancel = ["queued", "running", "paused", "blocked"].includes(step.status)
                  && activeStepJobs.length > 0 && activeStepJobs.every((job) => job.cancellable);
                const canRetry = UNSUCCESSFUL.has(step.status) && stepJobs.some((job) => UNSUCCESSFUL.has(job.status));
                return (
                  <li key={step.id}>
                    <span>
                      <strong>{steps.length > 1 ? `Output ${step.ordinal}` : "Progress"}</strong>
                      {stepJobs.length ? stepJobs.map((job) => <small key={job.id}>{progressText(job)}</small>)
                        : <small>{step.status.replaceAll("_", " ")}</small>}
                    </span>
                    {steps.length > 1 && canCancel && <button className="secondary compact-button" aria-label={`Cancel output ${step.ordinal}`}
                      onClick={() => onCancelStep(step.id)}>Cancel</button>}
                    {steps.length > 1 && canRetry && <button className="secondary compact-button" aria-label={`Retry output ${step.ordinal}`}
                      onClick={() => onRetryStep(step.id)}>Retry</button>}
                  </li>
                );
              })}
            </ol>
          </article>
        );
      })}
    </section>
  );
}
