import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { orderFamilies, servesCapability } from "./workflowFamilies";
import type {
  ChatWorkflowSelectionInput,
  ProjectWorkflowSelectionInput,
  WorkflowFamily,
  WorkflowFamilyVariant,
  WorkflowSelection,
  WorkflowSelectorCapability,
} from "./types";

const LEGACY_VALUE = "compatibility:legacy";
const REVISION_VALUE = "compatibility:revision";

/** What a variant's readiness means in words, since "setup_required" is not
 *  a sentence and the reason the server gives is often a code. */
function readinessNote(variant: WorkflowFamilyVariant): string | null {
  if (variant.readiness === "ready") return null;
  if (variant.readiness === "setup_required") {
    return variant.readiness_reason ?? "Needs files or nodes installed before it can run.";
  }
  return variant.readiness_reason ?? "Cannot run on this machine as configured.";
}

/** Choose which workflow answers one kind of request.
 *
 * The choice is a family rather than a revision: a family knows which of its
 * variants matches the operation being asked for, so picking one here does
 * not commit the user to a decision about text-to-image versus image-to-image
 * that they have no way to make yet.
 */
export function WorkflowSelector({
  scope,
  scopeId,
  capability,
  label,
}: {
  scope: "chat" | "project";
  scopeId: string;
  capability: WorkflowSelectorCapability;
  label: string;
}) {
  const client = useQueryClient();
  const families = useQuery({
    queryKey: ["workflow-families", capability],
    queryFn: () => api.workflowFamilies(capability),
  });
  const selections = useQuery({
    queryKey: [scope, scopeId, "workflow-selections"],
    queryFn: () =>
      scope === "chat"
        ? api.chatWorkflowSelections(scopeId)
        : api.projectWorkflowSelections(scopeId),
  });

  const choose = useMutation({
    mutationFn: (selection: ChatWorkflowSelectionInput | ProjectWorkflowSelectionInput) =>
      scope === "chat"
        ? api.setChatWorkflowSelection(
            scopeId,
            capability,
            selection as ChatWorkflowSelectionInput,
          )
        : api.setProjectWorkflowSelection(
            scopeId,
            capability,
            selection as ProjectWorkflowSelectionInput,
          ),
    onSuccess: () =>
      void client.invalidateQueries({ queryKey: [scope, scopeId, "workflow-selections"] }),
  });

  const current: WorkflowSelection | undefined = selections.data?.find(
    (selection) => selection.selector_capability === capability,
  );
  const available = (families.data ?? []).filter((family) => servesCapability(family, capability));
  const ordered = orderFamilies(available, capability);
  const selectedFamilyMissing = current?.mode === "family"
    && Boolean(current.workflow_family_id)
    && !ordered.some((family) => family.id === current.workflow_family_id);

  // "default" for a chat and "inherit" for a project are the same idea said
  // two ways: follow whatever the level above decided.
  const followMode = scope === "chat" ? "default" : "inherit";
  const value =
    current?.mode === "family" && current.workflow_family_id
      ? current.workflow_family_id
      : current?.mode === "automatic"
        ? "automatic"
        : current?.mode === "revision"
          ? REVISION_VALUE
          : current?.mode === "legacy"
            ? LEGACY_VALUE
            : followMode;

  return (
    <label className="workflow-selector">
      <span>{label}</span>
      <select
        value={value}
        disabled={choose.isPending || families.isLoading || selections.isLoading}
        onChange={(event) => {
          const next = event.target.value;
          if (next === REVISION_VALUE || next === LEGACY_VALUE) return;
          if (next === followMode) choose.mutate({ mode: followMode } as never);
          else if (next === "automatic") choose.mutate({ mode: "automatic" });
          else choose.mutate({ mode: "family", workflow_family_id: next });
        }}
      >
        <option value={followMode}>
          {scope === "chat" ? "Use the project's choice" : "Use the workspace default"}
        </option>
        <option value="automatic">Choose automatically</option>
        {current?.mode === "revision" && (
          <option value={REVISION_VALUE} disabled>Exact workflow revision (existing choice)</option>
        )}
        {current?.mode === "legacy" && (
          <option value={LEGACY_VALUE} disabled>Existing model setup</option>
        )}
        {selectedFamilyMissing && (
          <option value={current.workflow_family_id!} disabled>Selected workflow (unavailable)</option>
        )}
        {ordered.map((family) => (
          <option key={family.id} value={family.id}>
            {family.name}
            {family.compatibility ? " (existing setup)" : ""}
          </option>
        ))}
      </select>
      {current?.mode === "revision" && (
        <small>
          Pinned to one exact revision. Choosing a family here replaces that pin.
        </small>
      )}
      {current?.mode === "legacy" && (
        <small>
          Using the model previously configured here. Choosing a workflow replaces that choice.
        </small>
      )}
      {choose.error && <small role="alert">{(choose.error as Error).message}</small>}
      <WorkflowSelectorReadiness families={ordered} chosen={current?.workflow_family_id ?? null} />
    </label>
  );
}

/** Say when the chosen family cannot actually run, and why.
 *
 * A selector that lets you pick something unrunnable and only tells you at
 * generation time has moved the failure rather than prevented it.
 */
function WorkflowSelectorReadiness({
  families,
  chosen,
}: {
  families: WorkflowFamily[];
  chosen: string | null;
}) {
  const family = families.find((candidate) => candidate.id === chosen);
  if (!family) return null;
  const blocked = family.variants.filter((variant) => variant.readiness !== "ready");
  if (blocked.length === 0 || blocked.length < family.variants.length) return null;

  return (
    <small role="status" className="workflow-selector-blocked">
      {readinessNote(blocked[0])}
    </small>
  );
}
