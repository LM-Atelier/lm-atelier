import { useEffect, useRef, useState, type SetStateAction } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api, turnConfirmationForError } from "./api";
import { useTurnConfirmation } from "./useTurnConfirmation";
import type { ComposerProps } from "./chatComposerContracts";
import { ErrorCallout } from "./ErrorCallout";
import { ProfilePicker, ProfileReadStatus } from "./ProfilePicker";
import { useProfileLibrary } from "./useProfileLibrary";
import { GenerationPresetPicker } from "./GenerationPresetPicker";
import { useGenerationPresetLibrary } from "./useGenerationPresetLibrary";
import {
  buildPriorTurnEditPreviewRequest, confirmPriorTurnEditSubmission, initializePriorTurnEditDraft, preparePriorTurnEditSubmission, readPriorTurnEditDraft,
  removePriorTurnEditDraft, writePriorTurnEditDraft, priorTurnEditConfigurationSource, selectPriorTurnEditConfiguration,
  type PriorTurnEditDraft,
} from "./priorTurnEditDraft";
import { TurnEditor, type TurnEditorProps } from "./TurnEditor";
import { roleForMode } from "./viewHelpers";
import type { EngineRole, Message, PriorTurnEditAccepted, RoutingMode, TurnWorkflowSelectionInput, WorkflowSelection } from "./types";
import { workflowRevisionForTurn } from "./turnEditorContext";
import { operationForTurn } from "./turnWorkflow";
import { useWorkflowRevisionSchema } from "./useWorkflowRevisionSchema";
import { uniqueWorkflowRows, useWorkflowRevisionChoice, useWorkflowRevisionPages } from "./useWorkflowReadPages";
import { WorkflowReadPageControls } from "./WorkflowReadPageControls";
import { WorkflowFamilyBrowseControls } from "./WorkflowFamilyBrowseControls";
import { usePriorWorkflowFamilies } from "./usePriorWorkflowFamilies";
import { useWorkflowResolutionFamilies } from "./useWorkflowResolutionFamilies";
import { priorTurnImageWorkflowChoice, priorTurnSourceCanvasRevision } from "./priorTurnSourceCanvas";
import type { SourceFitPreviewContext } from "./useSourceFitCanvas";

type Props = Pick<ComposerProps, "chat" | "engines" | "maxMediaOutputsPerPlan"> & {
  messageId: string;
  onAccepted: (accepted: PriorTurnEditAccepted) => void;
  onClose: () => void;
  PromptHelper?: TurnEditorProps["PromptHelper"];
};

const ignore = () => {};

function contextMessages(draft: PriorTurnEditDraft): Message[] {
  return draft.source.context_messages.map((entry, index) => ({
    id: "edit-context-" + index, chat_id: draft.source.chat_id, parent_id: null,
    role: entry.role === "assistant" || entry.role === "system" || entry.role === "tool" ? entry.role : "user",
    status: "complete", created_at: "", updated_at: "",
    parts: [{ id: "edit-context-part-" + index, position: 0, type: "text", text: entry.content,
      artifact_id: null, metadata_json: {} }],
  }));
}

/** The edit a canvas is previewed within, once the workflow that would draw it is known.

Until then there is nothing exact to preview, so the control stays unavailable. */
function priorEditPreview(
  messageId: string, draft: PriorTurnEditDraft, revision: string | null | undefined,
): SourceFitPreviewContext | undefined {
  return revision ? { kind: "prior-edit", id: messageId, request: buildPriorTurnEditPreviewRequest(draft) } : undefined;
}

/** One source and one durable draft; every control changes only this request. */
export function PriorTurnEditor({ chat, messageId, engines,
  maxMediaOutputsPerPlan, onAccepted, onClose, PromptHelper }: Props) {
  const [draft, setDraft] = useState<PriorTurnEditDraft | null>(null);
  const current = useRef<PriorTurnEditDraft | null>(null);
  const [error, setError] = useState("");
  const [accepting, setAccepting] = useState(false);
  const [reload, setReload] = useState(0);
  const [confirmationDialog, confirmTurn] = useTurnConfirmation();
  const pending = useRef(false);
  const accepted = useRef(false);
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const restored = readPriorTurnEditDraft(chat.id, messageId);
        const value = restored ?? initializePriorTurnEditDraft(await api.getPriorTurnEditSource(messageId));
        if (!alive) return;
        current.current = value;
        setDraft(value);
        setError("");
      } catch (cause) {
        if (alive) setError(cause instanceof Error ? cause.message : "The source could not be loaded.");
      }
    })();
    return () => { alive = false; };
  }, [chat.id, messageId, reload]);

  const update = (change: SetStateAction<PriorTurnEditDraft>) => {
    if (!current.current || accepted.current) return;
    const next = typeof change === "function" ? change(current.current) : change;
    current.current = next;
    setDraft(next);
    try { writePriorTurnEditDraft(next); setError(""); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "The draft could not be saved."); }
  };
  const close = () => { if (!pending.current) onClose(); };
  const role = draft?.settingsRole ?? roleForMode(chat.routing_mode);
  const configurationSource = draft ? priorTurnEditConfigurationSource(draft) : undefined;
  const stepTarget = draft?.configurations && "stepId" in draft.configurations.target;
  const roleOverrides = stepTarget ? draft.configurations?.roles[role] : undefined;
  const effectivePreset = draft?.presetChoice.kind === "explicit" ? draft.presetChoice.value : roleOverrides?.preset_id;
  const presetLibrary = useGenerationPresetLibrary(role, [effectivePreset], Boolean(draft), false);
  const inheritedSettings = { ...(roleOverrides?.preset_id ? {} : configurationSource?.settings), ...roleOverrides?.settings };
  const capability = role === "chat" ? "chat" : role === "image" ? "image" : "video";
  const workflowChoice = draft?.workflowChoice;
  const selectedWorkflow = workflowChoice?.kind === "explicit" ? workflowChoice.value : roleOverrides?.workflow_selection;
  const workflowValue = workflowChoice?.kind !== "explicit" ? "inherit" : selectedWorkflow === null ? "current"
    : selectedWorkflow?.mode === "family" ? "family:" + selectedWorkflow.workflow_family_id
      : selectedWorkflow?.mode === "revision" ? "revision:" + selectedWorkflow.workflow_revision_id
        : selectedWorkflow?.mode ?? "inherit";
  const selection: WorkflowSelection | undefined = selectedWorkflow ? {
    selector_capability: selectedWorkflow.selector_capability, mode: selectedWorkflow.mode,
    workflow_family_id: selectedWorkflow.mode === "family" ? selectedWorkflow.workflow_family_id : null,
    workflow_revision_id: selectedWorkflow.mode === "revision" ? selectedWorkflow.workflow_revision_id : null,
    legacy_profile_id: null,
  } : undefined;
  const sameRole = Boolean(configurationSource);
  const inheritsSchema = Boolean(draft && workflowChoice?.kind === "inherit"
    && selectedWorkflow === undefined && sameRole);
  const imageChoice = priorTurnImageWorkflowChoice(draft);
  const resolution = useWorkflowResolutionFamilies([
    role !== "chat" && !inheritsSchema && (!selectedWorkflow || selectedWorkflow.mode === "family" || selectedWorkflow.mode === "default")
      ? { capability: role, operation: operationForTurn(role, Boolean(draft?.editor.attachments.length)),
        familyId: selectedWorkflow?.mode === "family" ? selectedWorkflow.workflow_family_id : null } : null,
    imageChoice?.mode === "family" || imageChoice?.mode === "default"
      ? { capability: "image", operation: "image_to_image",
        familyId: imageChoice.mode === "family" ? imageChoice.workflow_family_id : null } : null,
  ]);
  const families = usePriorWorkflowFamilies(selectedWorkflow?.mode === "family" ? selectedWorkflow.workflow_family_id : null);
  const missingFamily = workflowValue.startsWith("family:")
    && !families.families.some(family => "family:" + family.id === workflowValue);
  const revisionId = inheritsSchema ? null : selectedWorkflow?.mode === "revision"
    ? selectedWorkflow.workflow_revision_id
    : workflowRevisionForTurn(role === "chat" ? "text" : role,
        Boolean(draft?.editor.attachments.length), resolution.families, selection);
  // An explicit historical choice keeps that revision's own operation, as before.
  const expectedOperation = selectedWorkflow?.mode === "revision" ? undefined : role === "chat" ? "text"
    : operationForTurn(role, Boolean(draft?.editor.attachments.length));
  const schemaRead = useWorkflowRevisionSchema(revisionId, expectedOperation);
  const schema = inheritsSchema ? configurationSource?.workflow_schema : schemaRead.schema;
  const [revisionSearch, setRevisionSearch] = useState("");
  const revisionChoices = useWorkflowRevisionPages(role, revisionSearch);
  const selectedRevision = useWorkflowRevisionChoice(
    selectedWorkflow?.mode === "revision" ? selectedWorkflow.workflow_revision_id : "", role,
  );
  const revisionListError = revisionChoices.isFetchNextPageError ? null : revisionChoices.error;
  const visibleRevisions = uniqueWorkflowRows([
    ...(revisionListError ? [] : revisionChoices.data ?? []),
    ...(!selectedRevision.error && selectedRevision.data ? [selectedRevision.data] : []),
  ], revision => revision.revision_id)
    .filter((revision) => role === "chat" ? revision.operation === "text"
      : role === "video" ? revision.operation.includes("video")
        : revision.operation.includes("image") && !revision.operation.includes("video"));
  const missingPinnedRevision = workflowValue.startsWith("revision:")
    && !visibleRevisions.some((revision) => "revision:" + revision.revision_id === workflowValue);
  const profileChoice = draft?.profileChoice;
  const inheritsProfile = profileChoice?.kind === "inherit" && !Object.hasOwn(roleOverrides ?? {}, "profile_id") && sameRole;
  const hasProfileOverride = profileChoice?.kind === "explicit" || Object.hasOwn(roleOverrides ?? {}, "profile_id");
  const profileId = profileChoice?.kind === "explicit" ? profileChoice.value : roleOverrides?.profile_id;
  const profileLibrary = useProfileLibrary(role, inheritsProfile ? null : profileId, Boolean(draft), !inheritsProfile && profileId !== "__auto__");
  const visionChoice = draft?.visionProfileChoice;
  const visionId = visionChoice?.kind === "explicit" ? visionChoice.value : roleOverrides?.vision_profile_id;
  const visionLibrary = useProfileLibrary("chat", visionId, Boolean(draft) && role === "chat", false, "image");
  const profile = profileLibrary.identity.profile;
  const profileValues = inheritsProfile ? configurationSource?.profile_settings ?? {}
    : { ...profile?.load_settings_json, ...profile?.request_settings_json };
  const pickWorkflow = (value: string) => {
    if (value === "inherit") { update((value) => ({ ...value, workflowChoice: { kind: "inherit" } })); return; }
    let choice: TurnWorkflowSelectionInput | null;
    if (value === "current") choice = null;
    else if (value.startsWith("family:")) choice = { selector_capability: capability, mode: "family", workflow_family_id: value.slice(7) };
    else if (value.startsWith("revision:")) choice = { selector_capability: capability, mode: "revision", workflow_revision_id: value.slice(9) };
    else choice = { selector_capability: capability, mode: value === "automatic" ? "automatic" : "default" };
    update((value) => ({ ...value, workflowChoice: { kind: "explicit", value: choice } }));
  };
  const changeMode = (mode: RoutingMode) => update((value) => {
    const next = { ...value, editor: { ...value.editor, mode } };
    return mode === "auto" ? next : selectPriorTurnEditConfiguration(next, { role: roleForMode(mode) });
  });

  return <AccessibleDialog title="Queue edited version" eyebrow="Edit one turn" closeLabel="Close edited version"
    onClose={close}>
    {confirmationDialog}
    <p>The original and current generations will continue.</p>
    {error && <ErrorCallout message={error} />}
    {(revisionListError || selectedRevision.error || schemaRead.error) && <div>
      <ErrorCallout message="Workflow choices or settings could not be loaded." />
      <button type="button" onClick={() => {
        if (revisionChoices.error) void revisionChoices.refetch();
        if (selectedRevision.error) void selectedRevision.refetch();
        if (schemaRead.error) void schemaRead.retry();
      }}>Retry workflow reads</button>
    </div>}
    {!draft ? <div role="status">{error
      ? <button onClick={() => setReload((value) => value + 1)}>Try loading again</button> : "Loading original turn…"}</div>
      : <>
        <ProfilePicker library={profileLibrary} label="Model for this version" searchLabel="Search models for this version"
          disabled={accepting} value={draft.profileChoice.kind === "inherit" ? "inherit" : draft.profileChoice.value ?? "default"}
          selectedId={draft.profileChoice.kind === "explicit" ? draft.profileChoice.value : null}
          onChange={(selected) => update((value) => ({ ...value, profileChoice: selected === "inherit"
            ? { kind: "inherit" } : { kind: "explicit", value: selected === "default" ? null : selected } }))}>
          <option value="inherit">{stepTarget ? "Original model and role settings" : sameRole ? "Original model configuration" : "Current model selection"}</option>
          <option value="default">Current default</option>
          {draft.profileChoice.kind === "explicit" && draft.profileChoice.value === "__auto__" && <option value="__auto__">Automatic</option>}
        </ProfilePicker>
        {role === "chat" && <ProfilePicker library={visionLibrary} label="Vision model for this version" searchLabel="Search vision models for this version"
          disabled={accepting} value={draft.visionProfileChoice.kind === "inherit" ? "inherit" : draft.visionProfileChoice.value ?? "none"}
          selectedId={draft.visionProfileChoice.kind === "explicit" ? draft.visionProfileChoice.value : null}
          onChange={(selected) => update((value) => ({ ...value, visionProfileChoice: selected === "inherit"
            ? { kind: "inherit" } : { kind: "explicit", value: selected === "none" ? null : selected } }))}>
          <option value="inherit">Original vision configuration</option><option value="none">No vision model</option>
          {draft.visionProfileChoice.kind === "explicit" && draft.visionProfileChoice.value === "__auto__" && <option value="__auto__">Automatic</option>}
        </ProfilePicker>}
        <div aria-label="References for this version">
          {draft.source.references.filter((reference) => !(draft.removedReferenceSubjectIds ?? []).includes(reference.reference_subject_id))
            .map((reference) => <span key={reference.reference_subject_id}>{reference.subject_name}
              <button disabled={accepting} aria-label={"Remove reference " + reference.subject_name} onClick={() => update((value) => ({
                ...value, removedReferenceSubjectIds: [...(value.removedReferenceSubjectIds ?? []), reference.reference_subject_id],
              }))}>Remove</button>
            </span>)}
        </div>
        <TurnEditor chat={{ ...chat, routing_mode: draft.editor.mode, messages: contextMessages(draft), active_head_message_id: null }}
          engines={engines} maxMediaOutputsPerPlan={maxMediaOutputsPerPlan}
          stoppable={false} onStop={ignore} onStopAndSend={ignore} onSend={ignore}
          editorState={draft.editor} onEditorStateChange={(change) => update((value) => ({
            ...value, editor: typeof change === "function" ? change(value.editor) : change,
          }))} draft={draft.composer} onDraftChange={(change) => update((value) => ({
            ...value, composer: typeof change === "function" ? change(value.composer) : change,
          }))} onMode={changeMode} settingsRole={draft.settingsRole}
          onSettingsRole={(settingsRole) => update((value) => selectPriorTurnEditConfiguration(value, { role: settingsRole }))}
          settings={draft.settings} onSettings={(settings, changedKeys = []) => update((value) => ({
            ...value, settings, explicitSettingsKeys: [...new Set([...(value.explicitSettingsKeys ?? []), ...changedKeys])],
          }))}
          presetId={draft.presetChoice.kind === "explicit" ? draft.presetChoice.value
            : roleOverrides && Object.hasOwn(roleOverrides, "preset_id") ? roleOverrides.preset_id ?? null : configurationSource?.preset_id ?? null}
          onPreset={(presetId) => update((value) => ({ ...value, presetChoice: { kind: "explicit", value: presetId },
            settings: presetId ? { ...presetLibrary.choices.find((item) => item.id === presetId)?.settings_json } : value.settings }))}
          editSettings={{
            configurationControl: <label>Settings for this version<select aria-label="Settings for this version" disabled={accepting}
              value={draft.configurations && "stepId" in draft.configurations.target
                ? "step:" + draft.configurations.target.stepId : "role:" + draft.settingsRole}
              onChange={(event) => update((value) => selectPriorTurnEditConfiguration(value,
                event.target.value.startsWith("step:") ? { stepId: event.target.value.slice(5) }
                  : { role: event.target.value.slice(5) as EngineRole }))}>
              <option value="role:chat">All text steps</option><option value="role:image">All image steps</option>
              <option value="role:video">All video steps</option>
              {draft.source.steps?.map((step) => <option key={step.step_id} value={"step:" + step.step_id}>
                Step {step.ordinal + 1} · {step.settings_role === "chat" ? "text" : step.settings_role}
              </option>)}
            </select></label>,
            inheritedValues: sameRole && !effectivePreset
              ? { ...configurationSource?.resolved_settings, ...roleOverrides?.settings } : {},
            onRestore: () => update((value) => ({ ...value,
              settings: inheritedSettings, explicitSettingsKeys: [],
              presetChoice: { kind: "inherit" }, editor: { ...value.editor, templateSettings: null },
            })),
            presetControl: <><GenerationPresetPicker library={presetLibrary} label="Preset for this version"
              searchLabel="Search presets for this version" disabled={accepting}
              value={draft.presetChoice.kind === "inherit" ? "inherit" : draft.presetChoice.value ?? "none"}
              selectedId={draft.presetChoice.kind === "explicit" ? draft.presetChoice.value : null}
              onChange={(selected) => {
                update((value) => {
                  if (selected === "inherit") return { ...value, presetChoice: { kind: "inherit" },
                    settings: inheritedSettings, explicitSettingsKeys: [] };
                  if (selected === "none") return { ...value, presetChoice: { kind: "explicit", value: null } };
                  const settings = { ...presetLibrary.choices.find((item) => item.id === selected)?.settings_json };
                  return { ...value, presetChoice: { kind: "explicit", value: selected }, settings,
                    explicitSettingsKeys: Object.keys(settings) };
                });
              }}>
              <option value="inherit">{sameRole ? "Original preset settings" : "Current default preset"}</option>
              <option value="none">No preset</option>
            </GenerationPresetPicker>
              {effectivePreset && presetLibrary.pending && <p role="status">Loading selected preset…</p>}
              {(presetLibrary.error || presetLibrary.missing) && <ErrorCallout
                message={presetLibrary.error?.message ?? "The selected preset is unavailable."}
                action={<button type="button" onClick={presetLibrary.retry}>Retry selected preset</button>} />}
            </>,
          }}
          classificationSource={{ source_message_id: draft.source.source_user_message_id,
            source_run_id: draft.source.source_run_id, source_snapshot_sha256: draft.source.source_snapshot_sha256 }}
          contextMessages={contextMessages(draft)} contextVisualArtifacts={draft.source.context_visual_artifacts ?? []}
          profileSettingsUnavailable={!profileLibrary.identity.ready ? <ProfileReadStatus read={profileLibrary.identity} /> : undefined}
          profileValuesOverride={profileValues} workflowSchemaOverride={schema ?? null} workflowSelection={selection}
          sourceCanvasRevisionId={priorTurnSourceCanvasRevision(draft, resolution.families)}
          sourceFitPreviewContext={priorEditPreview(messageId, draft, priorTurnSourceCanvasRevision(draft, resolution.families))}
          workflowControl={<>
            <WorkflowFamilyBrowseControls browse={families.browse} label="workflow families" />
            {families.selected.isError && <p role="alert">Selected family could not be read. <button type="button"
              onClick={() => void families.selected.refetch()}>Retry selected family</button></p>}
            {resolution.error && <p role="alert">Selected workflow settings could not be resolved. <button type="button"
              onClick={() => void resolution.refetch()}>Retry workflow settings</button></p>}
            <label>Search workflow revisions<input type="search" maxLength={500} value={revisionSearch}
              onChange={(event) => setRevisionSearch(event.target.value)} /></label>
            <label>Workflow for this version<select aria-label="Workflow for this version" value={workflowValue}
            onChange={(event) => pickWorkflow(event.target.value)}>
            <option value="inherit">{sameRole ? "Original workflow" : "Current workflow selection"}</option>
            <option value="current">Current selection</option><option value="default">Default</option><option value="automatic">Auto</option>
            {families.families.map((family) =>
              <option key={family.id} value={"family:" + family.id}>{family.name}</option>)}
            {missingFamily && <option value={workflowValue}>
              {families.selected.isSuccess ? "Selected family (currently unavailable)" : "Selected family (details unavailable)"}
            </option>}
            {missingPinnedRevision && <option value={workflowValue} disabled>
              {selectedRevision.isPending ? "Selected workflow (loading details…)" : "Selected workflow (details unavailable)"}
            </option>}
            {visibleRevisions.map((revision) => <option key={revision.revision_id} value={"revision:" + revision.revision_id}>
                {revision.workflow_name} · version {revision.version}
              </option>)}
          </select></label>
          <WorkflowReadPageControls pages={revisionChoices} label="workflow revisions" />
          </>} PromptHelper={PromptHelper} submitLabel="Queue edited version"
          onAccept={async (submission) => {
            if (revisionId && (schemaRead.isLoading || schemaRead.error)) {
              throw new Error("Wait for the selected workflow settings to load, then try again.");
            }
            if ((hasProfileOverride && !profileLibrary.identity.ready) || !visionLibrary.identity.ready) {
              throw new Error("Wait for the selected model settings to load, or choose another model.");
            }
            if (effectivePreset && !presetLibrary.ready) {
              throw new Error("Wait for the selected preset to load, or choose another preset.");
            }
            if (!current.current) throw new Error("The source draft is unavailable.");
            const prepared = preparePriorTurnEditSubmission(current.current, submission);
            current.current = prepared.draft;
            setDraft(prepared.draft);
            pending.current = true;
            setAccepting(true);
            try {
              let result;
              try {
                result = await api.queueEditedMessage(prepared.draft.source.source_user_message_id, prepared.request);
              } catch (cause) {
                const requested = turnConfirmationForError(cause);
                if (prepared.request.confirm_media || !requested || !await confirmTurn(requested.confirmation)) throw cause;
                const confirmed = confirmPriorTurnEditSubmission(prepared.draft, requested.mode);
                current.current = confirmed.draft;
                setDraft(confirmed.draft);
                result = await api.queueEditedMessage(confirmed.draft.source.source_user_message_id, confirmed.request);
              }
              accepted.current = true;
              removePriorTurnEditDraft(chat.id, messageId, prepared.request.idempotency_key);
              onAccepted(result);
              onClose();
            } finally { pending.current = false; setAccepting(false); }
          }} />
      </>}
  </AccessibleDialog>;
}
