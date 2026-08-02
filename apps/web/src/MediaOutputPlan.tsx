import type { WorkPlan } from "./types";

export function MediaOutputPlan({
  plan,
  onCancelStep,
  onRetryStep,
}: {
  plan: WorkPlan;
  onCancelStep: (stepId: string) => void;
  onRetryStep: (stepId: string) => void;
}) {
  const steps = [...plan.steps].sort((left, right) => left.ordinal - right.ordinal);
  const ordered = plan.planner_version === "ordered-work-v1";
  const counts = steps.reduce<Record<string, number>>((current, step) => {
    current[step.status] = (current[step.status] ?? 0) + 1;
    return current;
  }, {});
  const summary = Object.entries(counts)
    .map(([status, count]) => `${count} ${status.replace("_", " ")}`)
    .join(" · ");
  return (
    <details className="media-output-plan">
      <summary>
        <span>{ordered ? `${steps.length}-step plan` : `${steps.length} media outputs`}</span>
        <small>{summary}</small>
      </summary>
      <ol>
        {steps.map((step) => {
          const cancellable = ["queued", "running", "paused", "blocked"].includes(step.status);
          const retryable = ["failed", "cancelled", "interrupted"].includes(step.status);
          const outputType = step.output_contract_json[0]?.type;
          const typeLabel = typeof outputType === "string"
            ? outputType[0].toUpperCase() + outputType.slice(1)
            : "Work";
          return (
            <li key={step.id}>
              <span>
                <strong>
                  {ordered ? `Step ${step.ordinal} · ${typeLabel}` : `Output ${step.ordinal}`}
                </strong>
                <small>{step.status.replace("_", " ")}</small>
                {step.error && <small className="error-text">{step.error}</small>}
              </span>
              {cancellable && (
                <button
                  className="secondary compact-button"
                  aria-label={`Cancel output ${step.ordinal}`}
                  onClick={() => onCancelStep(step.id)}
                >
                  Cancel
                </button>
              )}
              {retryable && (
                <button
                  className="secondary compact-button"
                  aria-label={`Retry output ${step.ordinal}`}
                  onClick={() => onRetryStep(step.id)}
                >
                  Retry
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </details>
  );
}
