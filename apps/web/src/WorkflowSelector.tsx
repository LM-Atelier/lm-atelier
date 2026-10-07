import { useId } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { readinessReason } from "./readinessReason";
import { useWorkflowFamilyChoices } from "./useWorkflowFamilyChoices";
import { useWorkflowSelectionSave } from "./useWorkflowSelectionSave";
import { WorkflowFamilyBrowseControls } from "./WorkflowFamilyBrowseControls";
import type {
  WorkflowFamily,
  WorkflowSelection,
  WorkflowSelectorCapability,
} from "./types";

const LEGACY_VALUE = "compatibility:legacy";
const REVISION_VALUE = "compatibility:revision";


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
  const selectorId = useId();
  const selections = useQuery({
    queryKey: [scope, scopeId, "workflow-selections"],
    queryFn: () =>
      scope === "chat"
        ? api.chatWorkflowSelections(scopeId)
        : api.projectWorkflowSelections(scopeId),
  });

  const current: WorkflowSelection | undefined = selections.data?.find(
    (selection) => selection.selector_capability === capability,
  );
  const families = useWorkflowFamilyChoices(
    capability, current?.mode === "family" ? current.workflow_family_id : null,
  );
  const ordered = families.families;

  const choose = useWorkflowSelectionSave({ kind: scope, id: scopeId, capability });

  // Neither read having arrived is not the same as an answer. Undefined data
  // fell straight through the value chain below: a failed families read
  // rendered "Selected workflow (unavailable)", telling someone their
  // configured workflow had been removed when one HTTP request had failed,
  // and a failed selections read read as "Use the project's choice" - an
  // inherited default presented as the confirmed current state. The natural
  // response to either is to pick something else, overwriting a setting that
  // was never broken.
  const readFailure = (families.error ?? selections.error) as Error | null;
  if (readFailure) {
    return (
      <div className="workflow-selector">
        <label htmlFor={selectorId}>{label}</label>
        <select id={selectorId} disabled value="">
          <option value="">Cannot read the current choice</option>
        </select>
        <small role="alert">
          {readFailure.message}
          <button
            className="secondary compact-button"
            onClick={() => {
              void families.refetch();
              void selections.refetch();
            }}
          >
            Try again
          </button>
        </small>
      </div>
    );
  }

  if (families.isLoading || selections.isLoading) {
    return <div className="workflow-selector">
      <label htmlFor={selectorId}>{label}</label>
      <select id={selectorId} disabled value="">
        <option value="">Loading current choice…</option>
      </select>
    </div>;
  }

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
    <div className="workflow-selector">
      <label htmlFor={selectorId}>{label}</label>
      <select
        id={selectorId}
        value={value}
        disabled={choose.saving || families.isLoading || selections.isLoading}
        onChange={(event) => {
          const next = event.target.value;
          if (next === REVISION_VALUE || next === LEGACY_VALUE) return;
          if (next === followMode) choose.choose({ mode: followMode });
          else if (next === "automatic") choose.choose({ mode: "automatic" });
          else choose.choose({ mode: "family", workflow_family_id: next });
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
      <WorkflowFamilyBrowseControls browse={families.browse} label={`${capability} workflows`} />
      <WorkflowSelectorReadiness families={ordered} chosen={current?.workflow_family_id ?? null} />
    </div>
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
  if ((family.ready_variant_count ?? family.variants.length - blocked.length) > 0
    || blocked.length === 0) return null;

  return (
    <small role="status" className="workflow-selector-blocked">
      {readinessReason(blocked[0])}
    </small>
  );
}
