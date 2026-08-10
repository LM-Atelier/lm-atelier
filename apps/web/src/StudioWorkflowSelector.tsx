import { useEffect, useId } from "react";
import type { ActiveChatWorkflowSelectionState } from "./useActiveChatWorkflowSelection";
import { useActiveChatWorkflowSelection } from "./useActiveChatWorkflowSelection";
import type { WorkflowFamily, WorkflowFamilyVariant } from "./types";

const LEGACY_VALUE = "compatibility:legacy";
const REVISION_VALUE = "compatibility:revision";

function readinessNote(variant: WorkflowFamilyVariant): string {
  if (variant.readiness === "setup_required") {
    return variant.readiness_reason ?? "Needs files or nodes installed before it can run.";
  }
  if (variant.readiness === "review_required") {
    return variant.readiness_reason ?? "Needs review before it can run.";
  }
  return variant.readiness_reason ?? "Cannot run on this machine as configured.";
}

function editVariants(family: WorkflowFamily | undefined): WorkflowFamilyVariant[] {
  return family?.variants.filter((variant) => variant.operation === "image_to_image") ?? [];
}

/** Why the confirmed Studio choice cannot currently edit an existing image. */
function studioWorkflowUnavailableReason(
  state: ActiveChatWorkflowSelectionState,
): string | null {
  if (state.kind === "loading") return "Loading the current workflow choice.";
  if (state.kind === "read-error") return "Cannot read the current workflow choice.";
  if (state.kind === "unresolved") return "The Image Studio workflow is unresolved.";
  if (state.saving) return "Saving the workflow choice.";
  if (!state.currentFamilyId) return null;

  const family = state.families.find((candidate) => candidate.id === state.currentFamilyId);
  if (!family) return "The selected workflow is no longer available.";
  const variants = editVariants(family);
  if (variants.length === 0) return "The selected workflow cannot edit an existing image.";
  const blocked = variants.filter((variant) => variant.readiness !== "ready");
  if (blocked.length === variants.length) return readinessNote(blocked[0]);
  return null;
}

/** Persist the workflow family on the hidden Studio chat. */
export function StudioWorkflowSelector({
  chatId,
  disabled,
  onAvailabilityChange,
  onSelectionChange,
}: {
  chatId: string;
  disabled: boolean;
  onAvailabilityChange: (reason: string | null) => void;
  onSelectionChange: () => void;
}) {
  const selectorId = useId();
  const state = useActiveChatWorkflowSelection(chatId, "image");
  const unavailableReason = studioWorkflowUnavailableReason(state);
  useEffect(() => {
    onAvailabilityChange(unavailableReason);
  }, [onAvailabilityChange, unavailableReason]);

  if (state.kind === "loading" || state.kind === "unresolved") {
    return (
      <div className="workflow-selector studio-workflow-selector">
        <label htmlFor={selectorId}>Editing workflow</label>
        <select id={selectorId} disabled value="">
          <option value="">Loading current choice…</option>
        </select>
      </div>
    );
  }
  if (state.kind === "read-error") {
    return (
      <div className="workflow-selector studio-workflow-selector">
        <label htmlFor={selectorId}>Editing workflow</label>
        <select id={selectorId} disabled value="">
          <option value="">Cannot read the current choice</option>
        </select>
        <small role="alert">
          {state.error.message}
          <button className="secondary compact-button" type="button" onClick={state.retry}>
            Try again
          </button>
        </small>
      </div>
    );
  }

  const editFamilies = state.families.filter((family) => editVariants(family).length > 0);
  const currentFamily = state.currentFamilyId
    ? state.families.find((family) => family.id === state.currentFamilyId)
    : undefined;
  const currentCanEdit = editVariants(currentFamily).length > 0;
  const unavailableCurrent = Boolean(
    state.currentFamilyId && (!currentFamily || !currentCanEdit),
  );
  const currentValue = state.choiceKind === "automatic"
    ? "automatic"
    : state.choiceKind === "explicit" && state.currentFamilyId
      ? state.currentFamilyId
      : state.current?.mode === "revision"
        ? REVISION_VALUE
        : state.current?.mode === "legacy"
          ? LEGACY_VALUE
          : "default";

  return (
    <div className="workflow-selector studio-workflow-selector">
      <label htmlFor={selectorId}>Editing workflow</label>
      <select
        id={selectorId}
        value={currentValue}
        disabled={disabled || state.saving}
        onChange={(event) => {
          const next = event.target.value;
          if (next === LEGACY_VALUE || next === REVISION_VALUE) return;
          onAvailabilityChange("Saving the workflow choice.");
          onSelectionChange();
          if (next === "default") state.choose({ mode: "default" });
          else if (next === "automatic") state.choose({ mode: "automatic" });
          else state.choose({ mode: "family", workflow_family_id: next });
        }}
      >
        <option value="default">Default</option>
        <option value="automatic">Auto</option>
        {state.current?.mode === "revision" && (
          <option value={REVISION_VALUE} disabled>Existing exact workflow</option>
        )}
        {state.current?.mode === "legacy" && (
          <option value={LEGACY_VALUE} disabled>Existing model setup</option>
        )}
        {unavailableCurrent && state.currentFamilyId && (
          <option value={state.currentFamilyId} disabled>
            Selected workflow cannot edit images
          </option>
        )}
        {editFamilies.map((family) => {
          const variants = editVariants(family);
          const blocked = variants.every((variant) => variant.readiness !== "ready");
          return (
            <option key={family.id} value={family.id} disabled={blocked}>
              {family.name}
              {family.compatibility ? " (existing setup)" : ""}
              {blocked ? " (not ready)" : ""}
            </option>
          );
        })}
      </select>
      <small>Choose which installed workflow edits this Studio session.</small>
      {state.current?.mode === "revision" && (
        <small>Choosing a workflow replaces the existing exact revision.</small>
      )}
      {state.current?.mode === "legacy" && (
        <small>Choosing a workflow replaces the existing model setup.</small>
      )}
      {state.saveError && <small role="alert">{state.saveError.message}</small>}
      {unavailableReason && !state.saving && (
        <small role="status" className="workflow-selector-blocked">{unavailableReason}</small>
      )}
    </div>
  );
}
