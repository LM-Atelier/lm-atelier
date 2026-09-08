import { useEffect, useRef, useState, type SetStateAction } from "react";
import { useQuery } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { api, turnConfirmationForError } from "./api";
import { useTurnConfirmation } from "./useTurnConfirmation";
import type { ComposerProps } from "./chatComposerContracts";
import { ErrorCallout } from "./ErrorCallout";
import {
  confirmPriorTurnEditSubmission, initializePriorTurnEditDraft, preparePriorTurnEditSubmission, readPriorTurnEditDraft,
  removePriorTurnEditDraft, writePriorTurnEditDraft, priorTurnEditConfigurationSource, selectPriorTurnEditConfiguration,
  type PriorTurnEditDraft,
} from "./priorTurnEditDraft";
import { TurnEditor, type TurnEditorProps } from "./TurnEditor";
import { roleForMode } from "./viewHelpers";
import type { EngineRole, Message, PriorTurnEditAccepted, RoutingMode, TurnWorkflowSelectionInput, WorkflowSelection } from "./types";
import { workflowSchemaForTurn } from "./turnEditorContext";

type Props = Pick<ComposerProps, "chat" | "engines" | "profiles" | "workflows" | "presets" | "maxMediaOutputsPerPlan"> & {
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

/** One source and one durable draft; every control changes only this request. */
export function PriorTurnEditor({ chat, messageId, engines, profiles, workflows, presets,
  maxMediaOutputsPerPlan, onAccepted, onClose, PromptHelper }: Props) {
  const [draft, setDraft] = useState<PriorTurnEditDraft | null>(null);
  const current = useRef<PriorTurnEditDraft | null>(null);
  const [error, setError] = useState("");
  const [accepting, setAccepting] = useState(false);
  const [reload, setReload] = useState(0);
  const [confirmationDialog, confirmTurn] = useTurnConfirmation();
  const pending = useRef(false);
  const accepted = useRef(false);
  const families = useQuery({ queryKey: ["workflow-families"], queryFn: () => api.workflowFamilies() });
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
  const schema = draft && workflowChoice?.kind === "inherit" && selectedWorkflow === undefined && sameRole ? configurationSource?.workflow_schema
    : selectedWorkflow?.mode === "revision"
      ? workflows.flatMap((workflow) => workflow.revisions).find((revision) => revision.id === selectedWorkflow.workflow_revision_id)?.input_schema_json
      : workflowSchemaForTurn(workflows, role === "chat" ? "text" : role, Boolean(draft?.editor.attachments.length),
        families.data ?? [], selection);
  const profileChoice = draft?.profileChoice;
  const profileId = profileChoice?.kind === "explicit" ? profileChoice.value
    : roleOverrides && Object.hasOwn(roleOverrides, "profile_id") ? roleOverrides.profile_id : configurationSource?.profile_id;
  const profile = profiles.find((item) => item.id === profileId);
  const profileValues = draft && profileChoice?.kind === "inherit" && !Object.hasOwn(roleOverrides ?? {}, "profile_id") && sameRole ? configurationSource?.profile_settings ?? {}
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
    {!draft ? <div role="status">{error
      ? <button onClick={() => setReload((value) => value + 1)}>Try loading again</button> : "Loading original turn…"}</div>
      : <>
        <label>Model for this version<select aria-label="Model for this version"
          disabled={accepting} value={draft.profileChoice.kind === "inherit" ? "inherit" : draft.profileChoice.value ?? "default"}
          onChange={(event) => update((value) => ({ ...value, profileChoice: event.target.value === "inherit"
            ? { kind: "inherit" } : { kind: "explicit", value: event.target.value === "default" ? null : event.target.value } }))}>
          <option value="inherit">{stepTarget ? "Original model and role settings" : sameRole ? "Original model configuration" : "Current model selection"}</option>
          <option value="default">Current default</option>
          {profiles.filter((item) => item.role === role).map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select></label>
        {role === "chat" && <label>Vision model for this version<select aria-label="Vision model for this version"
          disabled={accepting} value={draft.visionProfileChoice.kind === "inherit" ? "inherit" : draft.visionProfileChoice.value ?? "none"}
          onChange={(event) => update((value) => ({ ...value, visionProfileChoice: event.target.value === "inherit"
            ? { kind: "inherit" } : { kind: "explicit", value: event.target.value === "none" ? null : event.target.value } }))}>
          <option value="inherit">Original vision configuration</option><option value="none">No vision model</option>
          {profiles.filter((item) => item.role === "chat" && item.input_modalities?.includes("image"))
            .map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select></label>}
        <div aria-label="References for this version">
          {draft.source.references.filter((reference) => !(draft.removedReferenceSubjectIds ?? []).includes(reference.reference_subject_id))
            .map((reference) => <span key={reference.reference_subject_id}>{reference.subject_name}
              <button disabled={accepting} aria-label={"Remove reference " + reference.subject_name} onClick={() => update((value) => ({
                ...value, removedReferenceSubjectIds: [...(value.removedReferenceSubjectIds ?? []), reference.reference_subject_id],
              }))}>Remove</button>
            </span>)}
        </div>
        <TurnEditor chat={{ ...chat, routing_mode: draft.editor.mode, messages: contextMessages(draft), active_head_message_id: null }}
          engines={engines} profiles={profiles} workflows={workflows} presets={presets} maxMediaOutputsPerPlan={maxMediaOutputsPerPlan}
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
            settings: presetId ? { ...presets.find((item) => item.id === presetId)?.settings_json } : value.settings }))}
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
            presetControl: <label className="setting-row"><span><strong>Preset</strong></span>
              <select aria-label="Preset for this version" value={draft.presetChoice.kind === "inherit" ? "inherit" : draft.presetChoice.value ?? "none"}
                onChange={(event) => {
                  const selected = event.target.value;
                  update((value) => {
                    if (selected === "inherit") return { ...value, presetChoice: { kind: "inherit" },
                      settings: inheritedSettings, explicitSettingsKeys: [] };
                    if (selected === "none") return { ...value, presetChoice: { kind: "explicit", value: null } };
                    const settings = { ...presets.find((item) => item.id === selected)?.settings_json };
                    return { ...value, presetChoice: { kind: "explicit", value: selected }, settings,
                      explicitSettingsKeys: Object.keys(settings) };
                  });
                }}>
                <option value="inherit">{sameRole ? "Original preset settings" : "Current default preset"}</option>
                <option value="none">No preset</option>
                {presets.filter((item) => item.role === draft.settingsRole).map((item) =>
                  <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
            </label>,
          }}
          classificationSource={{ source_message_id: draft.source.source_user_message_id,
            source_run_id: draft.source.source_run_id, source_snapshot_sha256: draft.source.source_snapshot_sha256 }}
          contextMessages={contextMessages(draft)} contextVisualArtifacts={draft.source.context_visual_artifacts ?? []}
          profileValuesOverride={profileValues} workflowSchemaOverride={schema ?? null} workflowSelection={selection}
          workflowControl={<label>Workflow for this version<select aria-label="Workflow for this version" value={workflowValue}
            onChange={(event) => pickWorkflow(event.target.value)}>
            <option value="inherit">{sameRole ? "Original workflow" : "Current workflow selection"}</option>
            <option value="current">Current selection</option><option value="default">Default</option><option value="automatic">Auto</option>
            {(families.data ?? []).filter((family) => family.enabled && !family.archived).map((family) =>
              <option key={family.id} value={"family:" + family.id}>{family.name}</option>)}
            {workflows.filter((workflow) => role === "chat" ? workflow.operation === "text"
              : role === "video" ? workflow.operation.includes("video") : workflow.operation.includes("image") && !workflow.operation.includes("video"))
              .flatMap((workflow) => workflow.revisions.map((revision) =>
                <option key={revision.id} value={"revision:" + revision.id}>{workflow.name} · version {revision.version}</option>))}
          </select></label>} PromptHelper={PromptHelper} submitLabel="Queue edited version"
          onAccept={async (submission) => {
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
