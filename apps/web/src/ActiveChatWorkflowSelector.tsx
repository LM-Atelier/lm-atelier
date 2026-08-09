import { useId } from "react";
import { variantServesComposerCapability } from "./activeWorkflowCapability";
import { useActiveChatWorkflowSelection } from "./useActiveChatWorkflowSelection";
import type { RoutingMode, WorkflowFamilyVariant } from "./types";

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

export function ActiveChatWorkflowSelector({
  chatId,
  routingMode,
}: {
  chatId: string;
  routingMode: RoutingMode;
}) {
  const selectorId = useId();
  const state = useActiveChatWorkflowSelection(chatId, routingMode);
  if (state.kind === "unresolved") {
    return (
      <div className="workflow-selector" role="status">
        <span>Workflow</span>
        <small>Chosen after request classification</small>
      </div>
    );
  }
  if (state.kind === "loading") {
    return (
      <div className="workflow-selector">
        <label htmlFor={selectorId}>Workflow for this request type</label>
        <select id={selectorId} disabled value="">
          <option value="">Loading current choice…</option>
        </select>
      </div>
    );
  }
  if (state.kind === "read-error") {
    return (
      <div className="workflow-selector">
        <label htmlFor={selectorId}>Workflow for this request type</label>
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

  const currentValue = state.choiceKind === "automatic"
    ? "automatic"
    : state.choiceKind === "explicit" && state.currentFamilyId
      ? state.currentFamilyId
      : state.current?.mode === "revision"
        ? REVISION_VALUE
        : state.current?.mode === "legacy"
          ? LEGACY_VALUE
          : "default";
  const chosenFamily = state.families.find(
    (family) => family.id === state.currentFamilyId,
  );
  const applicableVariants = chosenFamily?.variants.filter(
    (variant) => variantServesComposerCapability(variant, state.capability),
  ) ?? [];
  const blockedVariants = applicableVariants.filter(
    (variant) => variant.readiness !== "ready",
  );
  const noApplicableVariant = Boolean(chosenFamily && applicableVariants.length === 0);
  const fullyBlocked = Boolean(
    applicableVariants.length > 0
    && blockedVariants.length === applicableVariants.length,
  );

  return (
    <div className="workflow-selector">
      <label htmlFor={selectorId}>Workflow for this request type</label>
      <select
        id={selectorId}
        value={currentValue}
        disabled={state.saving}
        onChange={(event) => {
          const next = event.target.value;
          if (next === LEGACY_VALUE || next === REVISION_VALUE) return;
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
        {state.selectedFamilyMissing && state.currentFamilyId && (
          <option value={state.currentFamilyId} disabled>Selected workflow (unavailable)</option>
        )}
        {state.families.map((family) => (
          <option key={family.id} value={family.id}>
            {family.name}{family.compatibility ? " (existing setup)" : ""}
          </option>
        ))}
      </select>
      {state.current?.mode === "revision" && (
        <small>Choosing a workflow replaces the existing exact revision.</small>
      )}
      {state.current?.mode === "legacy" && (
        <small>Choosing a workflow replaces the existing model setup.</small>
      )}
      {state.saveError && <small role="alert">{state.saveError.message}</small>}
      {noApplicableVariant && (
        <small role="status">This workflow has no {state.capability} variant.</small>
      )}
      {fullyBlocked && <small role="status">{readinessNote(blockedVariants[0])}</small>}
    </div>
  );
}
