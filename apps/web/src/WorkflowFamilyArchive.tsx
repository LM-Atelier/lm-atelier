import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { needsAttention, survivingWork } from "./workflowFamilyImpact";
import type { WorkflowFamily } from "./types";

/** Ask before archiving a family, with what it actually costs.
 *
 * The server refuses outright while work is genuinely in flight; this shows
 * that refusal rather than offering a button that cannot succeed.
 */
export function WorkflowFamilyArchive({
  family,
  onClose,
}: {
  family: WorkflowFamily;
  onClose: () => void;
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
      onClose();
    },
  });

  if (impact.isLoading) {
    return (
      <ConfirmDialog
        title={`Archive ${family.name}?`}
        question="Working out what this would affect…"
        confirmLabel="Archive"
        onConfirm={onClose}
        onCancel={onClose}
      />
    );
  }

  const found = impact.data;
  if (found?.archive_blocked) {
    return (
      <ConfirmDialog
        title={`${family.name} is still in use`}
        question="It cannot be archived while work is running against it. Let the queue drain, or stop the runs, and try again."
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
      title={`Archive ${family.name}?`}
      question="Archiving hides it from the selectors. It does not delete anything."
      detail={
        <div className="family-archive-impact">
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
      onConfirm={() => archive.mutate()}
      onCancel={onClose}
    />
  );
}
