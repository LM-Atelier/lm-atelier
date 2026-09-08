import { useCallback, useEffect, useRef, useState, type ReactNode, type SetStateAction, type ComponentType } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, CircleStop, Film, Image as ImageIcon, MessageSquare, Send, SlidersHorizontal, Sparkles, Wand2, Workflow as WorkflowIcon, X } from "lucide-react";
import { api } from "./api";
import { ActiveChatWorkflowSelector } from "./ActiveChatWorkflowSelector";
import { AttachControls } from "./AttachControls";
import { ComposerPromptTemplatesAction } from "./ComposerPromptTemplatesAction";
import { EditingStudio } from "./EditingStudio";
import { ErrorCallout } from "./ErrorCallout";
import { MessageField } from "./MessageField";
import { OutputCountControl } from "./OutputCountControl";
import { SettingsDrawer, type EditedVersionSettings } from "./SettingsDrawer";
import type { ComposerProps } from "./chatComposerContracts";
import { composerDraftWithText, detachedComposerDraft, promptSourceForTurn, type ComposerPromptSource } from "./composerPromptSource";
import { artifactSource, mediaOriginLabel } from "./messageMedia";
import { mediaOutputCountForTurn } from "./mediaOutputCount";
import { normalizeSettingsForFields, resolveCapabilitySettings, resolveWorkflowSettings } from "./settings";
import { activeBranchMessages, workflowSchemaForTurn } from "./turnEditorContext";
import { survivingMentions, turnReferences, type TurnReference } from "./mentionDraft";
import { useComposerUploads, type ComposerAttachment } from "./useComposerUploads";
import { useDraftClassification } from "./useDraftClassification";
import { drawerRoleView, roleForMode } from "./viewHelpers";
import { useTurnEditorState, type TurnEditorState } from "./useTurnEditorState";
export type { TurnEditorState } from "./useTurnEditorState";
import type { Artifact, ChatDetail, EngineCapabilities, EngineRole, Message, PriorTurnEditBinding, RoutingMode, Workflow, WorkflowSelection } from "./types";
export interface TurnEditorSubmission {
  requestId: string;
  text: string;
  mode: RoutingMode;
  /** Undefined inherits source inputs; an empty array intentionally removes them. */
  inputArtifactIds: string[] | undefined;
  settings: Record<string, unknown>;
  references: TurnReference[] | undefined;
  outputCount: number;
  promptSource?: ComposerPromptSource;
  presetId: string | null;
  settingsRole: EngineRole;
  workflowSelection?: WorkflowSelection;
}

export interface TurnEditorPromptHelperProps {
  sourceChat: ChatDetail;
  initialDraft: string;
  engines: EngineCapabilities[];
  workflows: Workflow[];
  editSourceArtifactIds?: string[];
  onAccept: (draft: string) => void;
  onClose: () => void;
}

export type TurnEditorProps = ComposerProps & {
  initialState?: Partial<TurnEditorState>;
  /** A prior-turn caller supplies source lineage instead of the active branch. */
  contextMessages?: Message[];
  classificationSource?: PriorTurnEditBinding;
  contextVisualArtifacts?: Artifact[];
  profileValuesOverride?: Record<string, unknown>;
  editSettings?: EditedVersionSettings;
  /** Supply a controlled workflow choice when editing one turn in isolation. */
  workflowControl?: ReactNode;
  workflowSelection?: WorkflowSelection;
  /** Null deliberately suppresses the current chat's workflow schema. */
  workflowSchemaOverride?: Record<string, unknown> | null;
  PromptHelper?: ComponentType<TurnEditorPromptHelperProps>;
  /** Resolve only after acceptance. Rejections leave the entire draft intact. */
  onAccept?: (submission: TurnEditorSubmission) => Promise<unknown>;
  submitLabel?: string;
} & (
  | { editorState: TurnEditorState; onEditorStateChange: (update: SetStateAction<TurnEditorState>) => void }
  | { editorState?: undefined; onEditorStateChange?: undefined }
);

function TurnEditorAttachments({ attachments, changeMode, onFocus, onAnimate, onRemove }: {
  attachments: ComposerAttachment[];
  changeMode: (mode: RoutingMode) => void;
  onFocus: () => void;
  onAnimate: () => void;
  onRemove: (id: string) => void;
}) {
  return (
    <div className="attachment-strip">
      {attachments.map((attachment) => {
        const source = attachment.artifact?.url || artifactSource(attachment.id)!;
        const name = attachment.artifact?.original_name || attachment.id;
        const label = mediaOriginLabel(attachment.origin, attachment.kind);
        return (
          <article className="attachment-card" key={attachment.id}>
            <a
              className="attachment-preview"
              href={source}
              target="_blank"
              rel="noreferrer"
              aria-label={`Preview ${name}`}
            >
              {attachment.kind === "image"
                ? <img src={source} alt="" />
                : <video src={source} muted preload="metadata" />}
            </a>
            <span className="attachment-summary">
              <strong>{label}</strong>
              <small title={name}>{name}</small>
            </span>
            <span className="attachment-actions">
              {attachment.kind === "image" && (
                <>
                  <button
                    className="attachment-edit"
                    aria-label="Edit attached image"
                    onClick={() => {
                      changeMode("image");
                      onFocus();
                    }}
                  >
                    Edit
                  </button>
                  <button
                    className="attachment-edit"
                    aria-label="Animate attached image"
                    onClick={() => {
                      changeMode("video");
                      onAnimate();
                      onFocus();
                    }}
                  >
                    Animate
                  </button>
                </>
              )}
              <button
                aria-label={`Remove ${label}: ${name}`}
                onClick={() => onRemove(attachment.id)}
              >
                <X size={12} />
              </button>
            </span>
          </article>
        );
      })}
    </div>
  );
}

export function TurnEditor({
  chat,
  engines,
  profiles,
  stoppable,
  settings,
  onSettings,
  settingsRole,
  onSettingsRole,
  presets,
  presetId,
  onPreset,
  onMode,
  onSend,
  onStop,
  onStopAndSend,
  maxMediaOutputsPerPlan,
  workflows,
  project,
  visualTarget,
  quoteTarget,
  draft,
  onDraftChange,
  initialState,
  editorState,
  onEditorStateChange,
  contextMessages,
  classificationSource,
  contextVisualArtifacts,
  profileValuesOverride,
  editSettings,
  workflowControl,
  workflowSelection,
  workflowSchemaOverride,
  PromptHelper,
  onAccept,
  submitLabel = "Send",
}: TurnEditorProps) {
  const text = draft.text;
  const setText = useCallback((next: string | ((current: string) => string)) => onDraftChange(
    (current) => composerDraftWithText(current, typeof next === "function" ? next(current.text) : next),
  ), [onDraftChange]);
  const detachPromptSource = useCallback(() => onDraftChange(detachedComposerDraft), [onDraftChange]);
  const { state, updateState, setOutputCount, changeMode, currentMode, setTemplateSettings, setAttachments } = useTurnEditorState(chat.routing_mode, onMode, initialState, editorState, onEditorStateChange);
  const { outputCount, mode, templateSettings, attachments } = state;
  const [accepting, setAccepting] = useState(false);
  const acceptancePending = useRef(false);
  const [acceptanceError, setAcceptanceError] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [promptHelperDraft, setPromptHelperDraft] = useState<string | null>(null);
  const [studioOpen, setStudioOpen] = useState(false);
  const addAttachment = (attachment: ComposerAttachment) => { detachPromptSource(); setAttachments((current) => [...current, attachment]); };
  const { uploading, uploadError, setUploadError, uploadFiles, uploadPastedImages } = useComposerUploads(addAttachment);
  const [dropActive, setDropActive] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const textInput = useRef<HTMLTextAreaElement>(null);
  const consumedVisualRequest = useRef<number | null>(null);
  useEffect(() => {
    if (!visualTarget || consumedVisualRequest.current === visualTarget.requestId) return;
    consumedVisualRequest.current = visualTarget.requestId;
    detachPromptSource();
    setAttachments((current) => {
      const additions = [visualTarget.attachment, ...(visualTarget.extraAttachments ?? [])]
        .filter((addition) => !current.some((item) => item.id === addition.id));
      return additions.length ? [...current, ...additions] : current;
    });
    if (visualTarget.mode) changeMode(visualTarget.mode);
    if (visualTarget.mode === "video") {
      window.setTimeout(() => {
        setText((current) => current.trim() ? current : "Animate this image");
      }, 0);
    }
    if (visualTarget.studio) {
      // After the attach renders, like the Animate prefill above.
      window.setTimeout(() => setStudioOpen(true), 0);
    }
    textInput.current?.focus();
  }, [visualTarget, changeMode, detachPromptSource, setText, setAttachments]);
  const consumedQuoteRequest = useRef<number | null>(null);
  useEffect(() => {
    if (!quoteTarget || consumedQuoteRequest.current === quoteTarget.requestId) return;
    consumedQuoteRequest.current = quoteTarget.requestId;
    const quoted = quoteTarget.text
      .trim()
      .split("\n")
      .map((line) => `> ${line}`)
      .join("\n");
    setText((current) => (current.trim() ? `${quoted}\n\n${current}` : `${quoted}\n\n`));
    textInput.current?.focus();
  }, [quoteTarget, setText]);
  const branchMessages = contextMessages ?? activeBranchMessages(chat);
  const priorVisual = Boolean(contextVisualArtifacts?.length) || branchMessages.some((message) =>
    message.parts.some((part) =>
      Boolean(part.artifact_id)
      && (part.type === "image" || part.type === "video")
      && part.metadata_json.preview !== true
    )
  );
  const priorImage = Boolean(contextVisualArtifacts?.some((item) => item.media_type.startsWith("image/"))) || branchMessages.some((message) =>
    message.parts.some((part) =>
      Boolean(part.artifact_id)
      && part.type === "image"
      && part.metadata_json.preview !== true
    )
  );
  const usePriorVisual = useDraftClassification(chat.id, text, mode, priorVisual, classificationSource);
  const editableImageAttached =
    attachments.some((attachment) => attachment.kind === "image")
    || (priorImage && usePriorVisual);
  const imageEdit = mode === "image" && editableImageAttached;
  // See drawerRoleView: the drawer follows the persisted routing mode and
  // the picked role, not the composer's local mode.
  const { drawerMode, drawerImageEdit } = drawerRoleView(onAccept ? "auto" : chat.routing_mode, settingsRole, editableImageAttached);
  const needsWorkflowSchema =
    mode === "image" || mode === "video" || drawerMode === "image" || drawerMode === "video";
  const families = useQuery({
    queryKey: ["workflow-families"],
    queryFn: () => api.workflowFamilies(),
    enabled: needsWorkflowSchema,
  });
  const selections = useQuery({
    queryKey: ["chat", chat?.id, "workflow-selections"],
    queryFn: () => api.chatWorkflowSelections(chat!.id),
    enabled: needsWorkflowSchema && workflowControl === undefined && Boolean(chat?.id),
  });
  const projectSelections = useQuery({ queryKey: ["project", project?.id, "workflow-selections"],
    queryFn: () => api.projectWorkflowSelections(project!.id),
    enabled: needsWorkflowSchema && workflowControl === undefined && Boolean(project?.id) });
  const imageProfile = profiles.find((profile) => profile.id === chat.active_image_profile_id)
    ?? profiles.find((profile) => profile.role === "image" && profile.is_default);
  const profileValues = profileValuesOverride ?? {
    ...(imageProfile?.load_settings_json ?? {}),
    ...(imageProfile?.request_settings_json ?? {}),
  };
  const workflowSchema = workflowSchemaOverride !== undefined ? workflowSchemaOverride ?? undefined : workflowSchemaForTurn(
    workflows,
    mode,
    attachments.length > 0 || usePriorVisual,
    families.data ?? [],
    workflowSelection ?? selections.data?.find((one) => one.selector_capability === mode),
    project ? projectSelections.data?.find((one) => one.selector_capability === mode) : null,
  );
  const drawerWorkflowSchema = workflowSchemaOverride !== undefined
    ? workflowSchemaOverride ?? undefined
    : drawerMode === mode
    ? workflowSchema
    : workflowSchemaForTurn(
        workflows,
        drawerMode,
        attachments.length > 0 || usePriorVisual,
        families.data ?? [],
        selections.data?.find((one) => one.selector_capability === drawerMode),
        project ? projectSelections.data?.find((one) => one.selector_capability === drawerMode) : null,
      );
  const clearAcceptedDraft = () => {
    setText("");
    updateState((current) => ({
      ...current, requestId: crypto.randomUUID(), submittedFingerprint: undefined,
      attachments: [], attachmentIntent: "replace", mentions: [], referenceIntent: "replace", outputCount: 1, templateSettings: null,
    }));
  };
  const submit = (stopCurrent = false) => {
    if (!text.trim() || uploading || acceptancePending.current) return;
    const selectedMode = currentMode();
    const role = roleForMode(selectedMode);
    const fields = resolveWorkflowSettings(resolveCapabilitySettings(engines.find((item) => item.roles.includes(role)), role), workflowSchema);
    const requestedOutputCount = mediaOutputCountForTurn(selectedMode, outputCount);
    const references = turnReferences(survivingMentions(text, state.mentions));
    const promptSource = promptSourceForTurn(draft, selectedMode, attachments.length, references.length, requestedOutputCount);
    const selectedSettings = onAccept ? { ...settings, ...templateSettings?.settings } : selectedMode === "auto" ? {} : normalizeSettingsForFields(
      templateSettings ? { ...settings, ...templateSettings.settings } : settings, fields,
    );
    if (!onAccept) {
      const dispatch = stopCurrent ? onStopAndSend : onSend;
      dispatch(text.trim(), selectedMode, attachments.map((item) => item.id), selectedSettings, references, requestedOutputCount, promptSource);
      clearAcceptedDraft();
      return;
    }
    const payload = {
      text, mode: selectedMode,
      inputArtifactIds: state.attachmentIntent === "inherit" ? undefined : attachments.map((item) => item.id),
      settings: selectedSettings,
      references: state.referenceIntent === "inherit" && references.length === state.mentions.length ? undefined : references,
      outputCount: requestedOutputCount ?? 1, promptSource, presetId, settingsRole, workflowSelection,
    };
    const fingerprint = JSON.stringify(payload);
    const requestId = state.submittedFingerprint && state.submittedFingerprint !== fingerprint
      ? crypto.randomUUID() : state.requestId;
    updateState((current) => ({ ...current, requestId, submittedFingerprint: fingerprint }));
    acceptancePending.current = true;
    setAccepting(true);
    setAcceptanceError("");
    void (async () => {
      try {
        await onAccept({ ...payload, requestId });
        clearAcceptedDraft();
      } catch (error) {
        setAcceptanceError(error instanceof Error ? error.message : "The request could not be accepted. Try again.");
      } finally {
        acceptancePending.current = false;
        setAccepting(false);
      }
    })();
  };

  return (
    <fieldset aria-label="Turn editor" disabled={accepting} style={{ border: 0, padding: 0, margin: 0, minWidth: 0 }}>
      <div
        className={`composer-wrap${dropActive ? " drop-active" : ""}`}
        style={onAccept ? { position: "relative", padding: 0 } : undefined}
        onDragOver={(event) => {
          if (acceptancePending.current) return;
          if (!Array.from(event.dataTransfer.types).includes("Files")) return;
          event.preventDefault();
          setDropActive(true);
        }}
        onDragLeave={(event) => {
          if (event.currentTarget.contains(event.relatedTarget as Node)) return;
          setDropActive(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          if (acceptancePending.current) return;
          setDropActive(false);
          const dropped = Array.from(event.dataTransfer.files);
          const files = dropped.filter(
            (file) => file.type.startsWith("image/") || file.type.startsWith("video/"),
          );
          setUploadError(
            files.length < dropped.length ? "Only images and videos can be attached." : "",
          );
          void uploadFiles(files);
        }}
      >
        {dropActive && <div className="drop-hint">Drop images or videos to attach</div>}
        {uploadError && <ErrorCallout message={uploadError} />}
        {acceptanceError && <ErrorCallout message={acceptanceError} />}
        {attachments.length > 0 && <TurnEditorAttachments attachments={attachments} changeMode={onAccept ? changeMode : onMode}
          onFocus={() => textInput.current?.focus()}
          onAnimate={() => { detachPromptSource(); setText((current) => current.trim() ? current : "Animate this image"); }}
          onRemove={(id) => setAttachments((items) => items.filter((item) => item.id !== id))} />}
        {templateSettings && (
          <div className="template-settings-chip">
            <span>{templateSettings.name} settings apply to this send</span>
            <button aria-label="Remove template settings" onClick={() => setTemplateSettings(null)}><X size={12} /></button>
          </div>
        )}
        {draft.promptSource && (
          <div className="template-settings-chip prompt-source-chip">
            <span>Prompt Library draft linked</span>
            <button aria-label="Remove Prompt Library source" onClick={detachPromptSource}><X size={12} /></button>
          </div>
        )}
        <div className="composer">
          <MessageField field={textInput} value={text} onChange={setText} onSubmit={submit} onMention={(mention) => { updateState((current) => ({ ...current, referenceIntent: "replace", mentions: [...current.mentions, mention] })); detachPromptSource(); }} onPasteFiles={(files) => { detachPromptSource(); void uploadPastedImages(files); }} />
          <div className="composer-tools">
            <div className="left-tools">
              <AttachControls disabled={uploading} onPickFile={() => fileInput.current?.click()} onAttach={addAttachment} />
              <input ref={fileInput} hidden multiple type="file" accept="image/*,video/*" onChange={(event) => { setUploadError(""); void uploadFiles(Array.from(event.target.files ?? [])); event.target.value = ""; }} />
              <button
                className="icon-button"
                onClick={() => setPromptHelperDraft(text.trim())}
                disabled={!text.trim() || !PromptHelper}
                aria-label="Open prompt workshop"
                title="Improve this prompt"
              >
                <Sparkles size={18} />
              </button>
              <ComposerPromptTemplatesAction chatId={chat.id} currentPrompt={text} maximum={maxMediaOutputsPerPlan} />
              <label className={`mode-select mode-${mode}`}>
                {mode === "auto" && <Sparkles size={15} />}
                {mode === "text" && <MessageSquare size={15} />}
                {mode === "image" && <ImageIcon size={15} />}
                {mode === "video" && <Film size={15} />}
                <select aria-label="Generation mode" value={mode} onChange={(event) => {
                  const nextMode = event.target.value as RoutingMode;
                  changeMode(nextMode);
                  if (nextMode !== "image") detachPromptSource();
                }}>
                  <option value="auto">Auto</option><option value="text">Text</option><option value="image">Image</option><option value="video">Video</option>
                </select>
                <ChevronDown size={13} />
              </label>
              <OutputCountControl mode={mode} maximum={maxMediaOutputsPerPlan} value={outputCount} onChange={(nextCount) => { setOutputCount(nextCount); if (nextCount > 1) detachPromptSource(); }} />
              <div className="composer-workflow-selector">
                <WorkflowIcon aria-hidden="true" size={15} />
                {workflowControl === undefined ? <ActiveChatWorkflowSelector chatId={chat.id} routingMode={mode} /> : workflowControl}
              </div>
              {imageEdit && <button className="icon-button" onClick={() => setStudioOpen(true)} aria-label="Open editing studio" title="One-click edits"><Wand2 size={18} /></button>}
              <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Turn settings"><SlidersHorizontal size={18} /></button>
            </div>
            <span className="composer-submit-actions">
              {stoppable && !onAccept && (
                <button
                  className="send-button stop"
                  onClick={onStop}
                  aria-label="Stop current response"
                  title="Stop current response"
                >
                  <CircleStop size={18} />
                </button>
              )}
              {stoppable && !onAccept && text.trim() && (
                <button
                  className="secondary stop-and-send"
                  onClick={() => submit(true)}
                  aria-label="Stop current response and send"
                >
                  Stop and send
                </button>
              )}
              <button
                className={onAccept ? "primary" : "send-button"}
                disabled={!text.trim() || uploading || accepting}
                onClick={() => submit()}
                aria-label={submitLabel}
              >
                <Send size={18} />{onAccept && <span>{submitLabel}</span>}
              </button>
            </span>
          </div>
        </div>
      </div>
      {studioOpen && <EditingStudio currentInstruction={text} onClose={() => setStudioOpen(false)} onPick={(instruction, template) => { setText(instruction); setTemplateSettings(Object.keys(template.settings_json).length ? { name: template.name, settings: template.settings_json } : null); setStudioOpen(false); window.setTimeout(() => textInput.current?.focus(), 0); }} imageCount={attachments.filter((item) => item.kind === "image").length} onApplyToEach={onAccept ? undefined : (instruction, template) => {
        const role = roleForMode("image");
        const engine = engines.find((item) => item.roles.includes(role));
        const fields = resolveWorkflowSettings(resolveCapabilitySettings(engine, role), workflowSchema);
        const merged = normalizeSettingsForFields({ ...settings, ...template.settings_json }, fields);
        // One ordinary edit turn per image: each queues, verifies, and retries
        // alone; the pending-work bound errs clearly rather than truncating.
        // No references: these are edits of the attached images themselves,
        // not a mention-driven turn, and the instruction was not composed in
        // the field that tracks mentions.
        for (const item of attachments.filter((entry) => entry.kind === "image")) onSend(instruction, "image", [item.id], merged, []);
        setAttachments([]); setText(""); setTemplateSettings(null); setStudioOpen(false);
      }} />}

      {promptHelperDraft !== null && PromptHelper && <PromptHelper
        sourceChat={chat} initialDraft={promptHelperDraft} engines={engines} workflows={workflows}
        // The helper has no lineage: only explicit attachments ground it.
        editSourceArtifactIds={imageEdit ? attachments.filter((item) => item.kind === "image").map((item) => item.id) : undefined}
        onAccept={(nextDraft) => {
          setText(nextDraft); setPromptHelperDraft(null);
          window.setTimeout(() => textInput.current?.focus(), 0);
        }}
        onClose={() => setPromptHelperDraft(null)}
      />}
      <SettingsDrawer
        editSettings={editSettings}
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        mode={onAccept ? mode : chat.routing_mode}
        role={settingsRole}
        onRole={onSettingsRole}
        engines={engines}
        values={settings}
        onValues={onSettings}
        presets={presets}
        presetId={presetId}
        onPreset={onPreset}
        workflowSchema={drawerWorkflowSchema}
        inheritedValues={onAccept ? undefined : project?.generation_settings_json?.[settingsRole]}
        inheritedPresetId={onAccept ? undefined : project?.generation_preset_ids_json?.[settingsRole]}
        profileValues={profileValues}
        imageEdit={drawerImageEdit}
        imageEditPrompt={text}
      />
    </fieldset>
  );
}
