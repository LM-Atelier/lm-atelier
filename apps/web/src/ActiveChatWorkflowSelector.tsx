import { WorkflowChoiceDropdown, type WorkflowDropdownOption } from "./WorkflowChoiceDropdown";
import { useId } from "react";
import { variantServesComposerCapability } from "./activeWorkflowCapability";
import { useActiveChatWorkflowSelection } from "./useActiveChatWorkflowSelection";
import { readinessReason } from "./readinessReason";
import type { RoutingMode } from "./types";

const LEGACY_VALUE = "compatibility:legacy";
const REVISION_VALUE = "compatibility:revision";
const idleBrowse = { search: "", setSearch: () => {}, pages: { error: null, isPending: false,
  isFetchingNextPage: false, isFetchNextPageError: false, hasNextPage: false,
  fetchNextPage: () => Promise.resolve(), refetch: () => Promise.resolve() } };


export function ActiveChatWorkflowSelector({
  chatId,
  routingMode,
  label = "Workflow for this request type",
}: {
  chatId: string;
  routingMode: RoutingMode;
  label?: string;
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
        <label htmlFor={selectorId}>{label}</label>
        <WorkflowChoiceDropdown id={selectorId} label={label} browseLabel={`${state.capability} workflows`}
          value="" options={[]} browse={idleBrowse} saving={false} unavailableText="Loading current choice…" onChange={() => {}} />
      </div>
    );
  }
  if (state.kind === "read-error") {
    return (
      <div className="workflow-selector">
        <label htmlFor={selectorId}>{label}</label>
        <WorkflowChoiceDropdown id={selectorId} label={label} browseLabel={`${state.capability} workflows`}
          value="" options={[]} browse={idleBrowse} saving={false} unavailableText="Cannot read the current choice" onChange={() => {}} />
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
  const noApplicableVariant = Boolean(chosenFamily && (chosenFamily.variant_count ?? applicableVariants.length) === 0);
  const fullyBlocked = Boolean(
    applicableVariants.length > 0
    && (chosenFamily?.ready_variant_count ?? applicableVariants.length - blockedVariants.length) === 0,
  );
  const options: WorkflowDropdownOption[] = [
    { value: "default", label: "Default" }, { value: "automatic", label: "Auto" },
    ...(state.current?.mode === "revision" ? [{ value: REVISION_VALUE, label: "Existing exact workflow", disabled: true }] : []),
    ...(state.current?.mode === "legacy" ? [{ value: LEGACY_VALUE, label: "Existing model setup", disabled: true }] : []),
    ...(state.selectedFamilyMissing && state.currentFamilyId
      ? [{ value: state.currentFamilyId, label: "Selected workflow (unavailable)", disabled: true }] : []),
    ...state.families.map(family => ({ value: family.id,
      label: family.name + (family.compatibility ? " (existing setup)" : ""),
      searchResult: family.id !== state.currentFamilyId })),
  ];

  return (
    <div className="workflow-selector">
      <label htmlFor={selectorId}>{label}</label>
      <WorkflowChoiceDropdown id={selectorId} label={label} value={currentValue} options={options}
        browse={state.browse} browseLabel={`${state.capability} workflows`} saving={state.saving}
        onChange={(next) => {
          if (next === LEGACY_VALUE || next === REVISION_VALUE) return;
          if (next === "default") state.choose({ mode: "default" });
          else if (next === "automatic") state.choose({ mode: "automatic" });
          else state.choose({ mode: "family", workflow_family_id: next });
        }} />
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
      {fullyBlocked && <small role="status">{readinessReason(blockedVariants[0])}</small>}
    </div>
  );
}
