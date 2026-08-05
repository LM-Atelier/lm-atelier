import { LoaderCircle } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { SetupWizard } from "./SetupWizard";
import type { SetupReadinessReport, SetupRoleReadiness } from "./types";

/** Everything the app shows about local setup: the wait, the failure, the wizard.
 *
 * One surface rather than two conditionals in the shell, because the three
 * states are the same question answered at different moments - whether this
 * machine is ready - and dismissing any of them means the same thing.
 */
export function SetupSurface({
  open,
  visible,
  report,
  error,
  onRetry,
  onDismiss,
  onClose,
  onOpenModels,
  onOpenWorkflows,
}: {
  open: boolean | null;
  visible: boolean;
  report: SetupReadinessReport | undefined;
  error: Error | null;
  onRetry: () => void;
  /** Dismissed for this session: the shell remembers, not this component. */
  onDismiss: () => void;
  onClose: () => void;
  onOpenModels: (role: SetupRoleReadiness["role"]) => void;
  onOpenWorkflows: () => void;
}) {
  if (visible && report) {
    return (
      <SetupWizard
        report={report}
        onClose={onDismiss}
        onOpenModels={onOpenModels}
        onOpenWorkflows={onOpenWorkflows}
      />
    );
  }
  if (open !== true || report) return null;
  return (
    <AccessibleDialog
      title="Checking local setup"
      eyebrow="Local models"
      closeLabel="Close setup"
      onClose={onDismiss}
      className="setup-wizard"
    >
      {error ? (
        <ErrorCallout
          message={error.message}
          action={
            <button className="secondary compact-button" onClick={onRetry}>
              Retry
            </button>
          }
        />
      ) : (
        <div className="submission-progress">
          <LoaderCircle size={17} />
          <span>Checking models and runtimes…</span>
        </div>
      )}
      <footer>
        <button className="secondary" onClick={onClose}>
          Not now
        </button>
      </footer>
    </AccessibleDialog>
  );
}
