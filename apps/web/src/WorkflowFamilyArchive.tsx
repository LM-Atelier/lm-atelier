import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { needsAttention, survivingWork } from "./workflowFamilyImpact";
import type { WorkflowFamily } from "./types";

/** Ask before archiving a family, with what it actually costs.
 *
 * The server requires selections and defaults to move before archiving.
 * Accepted work and immutable revisions survive the change.
 */
export function WorkflowFamilyArchive({
  family,
  onClose,
  onArchived,
}: {
  family: WorkflowFamily;
  onClose: () => void;
  onArchived?: () => void;
}) {
  const client = useQueryClient();
  const impact = useQuery({
    queryKey: ["workflow-family", family.id, "removal-impact"],
    queryFn: () => api.workflowFamilyRemovalImpact(family.id),
  });
  const archive = useMutation({
    mutationFn: () => api.updateWorkflowFamily(family.id, { archived: true }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["workflow-families"] });
      onArchived?.();
      onClose();
    },
  });

  if (impact.isLoading) {
    return (
      <ConfirmDialog
        tone="action"
        title={`Archive ${family.name}?`}
        question="Working out what this would affect…"
        confirmLabel="Archive"
        confirmDisabled
        onConfirm={onClose}
        onCancel={onClose}
      />
    );
  }

  if (impact.error) {
    // Offering the action anyway would present "nothing will be affected" as
    // a finding, when the truth is that nobody managed to look. Archiving is
    // reversible, but a decision taken on invented evidence is not a decision.
    return (
      <ConfirmDialog
        tone="action"
        title={`Archive ${family.name}?`}
        question="What this would affect could not be read, so it is not being offered yet."
        detail={
          <div className="callout error" role="alert">
            <span>{(impact.error as Error).message}</span>
          </div>
        }
        confirmLabel="Try again"
        confirmDisabled={impact.isFetching}
        onConfirm={() => void impact.refetch()}
        onCancel={onClose}
      />
    );
  }

  const found = impact.data;
  if (found?.archive_blocked) {
    return (
      <ConfirmDialog
        tone="action"
        title={`${family.name} is still in use`}
        question="This family is selected by a chat or project, or is set as a default. Choose another workflow in those places before archiving it."
        confirmLabel="Close"
        onConfirm={onClose}
        onCancel={onClose}
      />
    );
  }

  const kept = found ? survivingWork(found) : [];
  const attention = found ? needsAttention(found) : [];

  return (
    <ConfirmDialog
        tone="action"
      title={`Archive ${family.name}?`}
      question="Archiving hides it from the selectors. It does not delete anything."
      detail={
        <div className="family-archive-impact">
          {/* Success closes this dialog, so a refusal used to leave it sitting
              open and unchanged - indistinguishable from a press that missed. */}
          {archive.error && (
            <div className="callout error" role="alert">
              <span>{archive.error.message}</span>
            </div>
          )}
          {attention.length > 0 && (
            <section>
              <h4>What changes</h4>
              <ul>
                {attention.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </section>
          )}
          {kept.length > 0 && (
            <section>
              <h4>What is kept</h4>
              <ul>
                {kept.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </section>
          )}
        </div>
      }
      confirmLabel="Archive it"
      confirmDisabled={archive.isPending}
      onConfirm={() => archive.mutate()}
      onCancel={onClose}
    />
  );
}
