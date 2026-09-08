import type { ComposerDraft, ComposerPromptSource } from "./composerPromptSource";
import { artifactOrigin } from "./messageMedia";
import { survivingMentions, turnReferences } from "./mentionDraft";
import type { TurnEditorSubmission } from "./TurnEditor";
import type { TurnEditorState } from "./useTurnEditorState";
import type { EngineRole, RoutingMode, PriorTurnEditConfiguration, PriorTurnEditRequest, PriorTurnEditSource, TurnReferenceInput, TurnRoleOverrides, TurnWorkflowSelectionInput } from "./types";

export type PriorTurnEditChoice<T> = { kind: "inherit" } | { kind: "explicit"; value: T | null };
export type PriorTurnEditStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;
export type PriorTurnConfigurationTarget = { role: EngineRole } | { stepId: string };

export interface PriorTurnEditDraft {
  version: 1;
  source: PriorTurnEditSource;
  composer: ComposerDraft;
  editor: TurnEditorState;
  settings: Record<string, unknown>;
  /** Controls record deliberate same-value reselection here, independently of value diffs. */
  explicitSettingsKeys?: string[];
  /** Explicit reference removal is independent of text edits. */
  removedReferenceSubjectIds?: string[];
  settingsRole: EngineRole;
  presetChoice: PriorTurnEditChoice<string>;
  workflowChoice: PriorTurnEditChoice<TurnWorkflowSelectionInput>;
  profileChoice: PriorTurnEditChoice<string>;
  visionProfileChoice: PriorTurnEditChoice<string>;
  configurations?: { target: PriorTurnConfigurationTarget; roles: Partial<Record<EngineRole, TurnRoleOverrides>>; steps: Record<string, TurnRoleOverrides> };
  pending?: { fingerprint: string; editableFingerprint?: string; request: PriorTurnEditRequest };
}

const PREFIX = "lm-atelier:prior-turn-edit:v1:";
const STORAGE_ERROR = "Could not save the edited version. Check browser storage and try again.";
const READ_ERROR = "Could not restore the edited version. Check browser storage and try again.";

function key(chatId: string, messageId: string): string {
  return `${PREFIX}${encodeURIComponent(chatId)}:${encodeURIComponent(messageId)}`;
}

function snapshot<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
      .map(([name, item]) => [name, canonical(item)]));
  }
  return value;
}

/** Copy source values once; later query refreshes must not reset this draft. */
export function initializePriorTurnEditDraft(source: PriorTurnEditSource): PriorTurnEditDraft {
  const saved = snapshot(source);
  const role = saved.settings_role;
  if (role !== "chat" && role !== "image" && role !== "video") throw new Error("The source settings role is unavailable.");
  const step = (saved.steps?.length ?? 0) > 1 ? saved.steps?.find((item) => item.source_run_id === saved.source_run_id) : undefined;
  return {
    version: 1,
    source: saved,
    composer: { text: saved.text, promptSource: saved.prompt_source as unknown as ComposerPromptSource | null },
    editor: {
      requestId: crypto.randomUUID(), mode: saved.original_mode ?? saved.mode,
      attachments: saved.input_artifact_ids.map((id) => {
        const artifact = saved.input_artifacts.find((item) => item.id === id);
        if (!artifact) throw new Error("A source attachment is unavailable.");
        return { id, artifact, kind: artifact.media_type.startsWith("video/") ? "video" : "image", origin: artifactOrigin(artifact) ?? "uploaded" };
      }),
      attachmentIntent: "inherit",
      mentions: saved.references.filter((reference) => reference.source === "mention")
        .map((reference) => ({ referenceSubjectId: reference.reference_subject_id, mentionSlug: reference.mention_slug })),
      referenceIntent: "inherit", outputCount: saved.output_count, templateSettings: null,
    },
    settings: snapshot(step?.settings ?? saved.settings), explicitSettingsKeys: [], removedReferenceSubjectIds: [], settingsRole: role,
    presetChoice: { kind: "inherit" }, workflowChoice: { kind: "inherit" },
    profileChoice: { kind: "inherit" }, visionProfileChoice: { kind: "inherit" },
    configurations: { target: step ? { stepId: step.step_id } : { role }, roles: {}, steps: {} },
  };
}

export function priorTurnEditConfigurationSource(draft: PriorTurnEditDraft, target = draft.configurations?.target ?? { role: draft.settingsRole }): PriorTurnEditConfiguration | undefined {
  if ("stepId" in target) {
    const step = draft.source.steps?.find((item) => item.step_id === target.stepId);
    if (!step) throw new Error("The source step is unavailable.");
    return step;
  }
  const steps = draft.source.steps?.filter((item) => item.settings_role === target.role) ?? [];
  if (steps.length > 1) return undefined;
  return steps[0] ?? (draft.source.settings_role === target.role ? draft.source : undefined);
}

function settingsDifference(values: Record<string, unknown>, inherited: Record<string, unknown>, explicitKeys: string[] = []): Record<string, unknown> {
  return Object.fromEntries(Object.entries(values).filter(([name, value]) => explicitKeys.includes(name)
    || JSON.stringify(canonical(value)) !== JSON.stringify(canonical(inherited[name]))));
}

function storeConfiguration(draft: PriorTurnEditDraft): PriorTurnEditDraft {
  const configurations = snapshot(draft.configurations ?? { target: { role: draft.settingsRole }, roles: {}, steps: {} });
  const inherited = priorTurnEditConfigurationSource(draft, configurations.target);
  const roleValues = "stepId" in configurations.target ? configurations.roles[draft.settingsRole]?.settings : undefined;
  const settings = settingsDifference(draft.settings, { ...inherited?.settings, ...roleValues }, draft.explicitSettingsKeys);
  const changes: TurnRoleOverrides = {};
  if (Object.keys(settings).length) changes.settings = settings;
  if (draft.presetChoice.kind === "explicit") changes.preset_id = draft.presetChoice.value;
  if (draft.profileChoice.kind === "explicit") changes.profile_id = draft.profileChoice.value;
  if (draft.visionProfileChoice.kind === "explicit") changes.vision_profile_id = draft.visionProfileChoice.value;
  if (draft.workflowChoice.kind === "explicit") changes.workflow_selection = draft.workflowChoice.value;
  if ("stepId" in configurations.target) {
    if (Object.keys(changes).length) configurations.steps[configurations.target.stepId] = changes;
    else delete configurations.steps[configurations.target.stepId];
  } else if (Object.keys(changes).length) configurations.roles[configurations.target.role] = changes;
  else delete configurations.roles[configurations.target.role];
  return { ...draft, configurations };
}

function explicitChoice<T>(values: object, name: string, value: T | null | undefined): PriorTurnEditChoice<T> {
  return Object.hasOwn(values, name) ? { kind: "explicit", value: value ?? null } : { kind: "inherit" };
}

/** Navigation saves only intent; it never chooses a new request key or clears template settings. */
export function selectPriorTurnEditConfiguration(draft: PriorTurnEditDraft, target: PriorTurnConfigurationTarget): PriorTurnEditDraft {
  const next = storeConfiguration(draft);
  const inherited = priorTurnEditConfigurationSource(next, target);
  const role = "role" in target ? target.role : inherited?.settings_role;
  if (role !== "chat" && role !== "image" && role !== "video") throw new Error("The source settings role is unavailable.");
  const roles = next.configurations!.roles;
  const changes = ("stepId" in target ? next.configurations!.steps[target.stepId] : roles[role]) ?? {};
  const preset = Object.hasOwn(changes, "preset_id") ? changes.preset_id
    : "stepId" in target ? roles[role]?.preset_id : undefined;
  return { ...next, configurations: { ...next.configurations!, target }, settingsRole: role,
    settings: snapshot({ ...(preset ? {} : inherited?.settings), ...("stepId" in target ? roles[role]?.settings : {}), ...changes.settings }),
    explicitSettingsKeys: Object.keys(changes.settings ?? {}),
    presetChoice: explicitChoice(changes, "preset_id", changes.preset_id),
    profileChoice: explicitChoice(changes, "profile_id", changes.profile_id),
    visionProfileChoice: explicitChoice(changes, "vision_profile_id", changes.vision_profile_id),
    workflowChoice: explicitChoice(changes, "workflow_selection", changes.workflow_selection),
  };
}

/** Selection intent is independent of displayed source IDs, including same-ID reselection. */
function buildLegacyPriorTurnEditRequest(draft: PriorTurnEditDraft, submission?: TurnEditorSubmission): PriorTurnEditRequest {
  const { source, editor } = draft;
  const mentions = survivingMentions(submission?.text ?? draft.composer.text, editor.mentions);
  const references = submission ? submission.references : editor.referenceIntent === "inherit" && mentions.length === editor.mentions.length
    ? undefined : turnReferences(mentions);
  const displayedSettings = submission?.settings ?? { ...draft.settings, ...editor.templateSettings?.settings };
  const explicitKeys = new Set(draft.explicitSettingsKeys ?? []);
  const settings = Object.fromEntries(Object.entries(displayedSettings).filter(([name, value]) => explicitKeys.has(name)
    || JSON.stringify(canonical(value)) !== JSON.stringify(canonical(source.settings[name]))));
  const request: PriorTurnEditRequest = {
    text: submission?.text ?? draft.composer.text,
    idempotency_key: draft.pending?.request.idempotency_key ?? editor.requestId,
    source_run_id: source.source_run_id,
    source_snapshot_sha256: source.source_snapshot_sha256,
    mode: submission?.mode ?? editor.mode,
    settings,
    output_count: submission?.outputCount ?? editor.outputCount,
  };
  const inputs = submission ? submission.inputArtifactIds : editor.attachmentIntent === "inherit" ? undefined : editor.attachments.map((item) => item.id);
  if (inputs !== undefined) request.input_artifact_ids = inputs;
  const removed = new Set(draft.removedReferenceSubjectIds ?? []);
  if (references !== undefined || removed.size > 0) {
    const selected = references ?? turnReferences(mentions);
    const selectedIds = new Set(selected.map((reference) => reference.reference_subject_id));
    const sourceIds = new Set(source.references.map((reference) => reference.reference_subject_id));
    request.references = source.references
      .filter((reference) => !removed.has(reference.reference_subject_id)
        && (reference.source !== "mention" || selectedIds.has(reference.reference_subject_id)))
      .map((reference): TurnReferenceInput => ({
        reference_subject_id: reference.reference_subject_id,
        source: reference.source === "mention" || reference.source === "picker" ? reference.source : "inherited_context",
        role: reference.role, strength: reference.strength, selected_asset_ids: reference.reference_asset_ids_json,
      }));
    request.references.push(...selected.filter((reference) => !sourceIds.has(reference.reference_subject_id)
      && !removed.has(reference.reference_subject_id)));
  }
  const promptSource = submission ? submission.promptSource ?? null : draft.composer.promptSource;
  // Omission preserves the accepted binding; null records deliberate detachment.
  if (JSON.stringify(promptSource) !== JSON.stringify(source.prompt_source)) request.prompt_source = promptSource;
  if (draft.presetChoice.kind === "explicit") request.preset_id = draft.presetChoice.value;
  if (draft.workflowChoice.kind === "explicit") request.workflow_selection = draft.workflowChoice.value;
  if (draft.profileChoice.kind === "explicit") request.profile_id = draft.profileChoice.value;
  if (draft.visionProfileChoice.kind === "explicit") request.vision_profile_id = draft.visionProfileChoice.value;
  return snapshot(request);
}

export function buildPriorTurnEditRequest(draft: PriorTurnEditDraft, submission?: TurnEditorSubmission): PriorTurnEditRequest {
  if (!draft.configurations) return buildLegacyPriorTurnEditRequest(draft, submission);
  const saved = storeConfiguration(draft);
  const mode = submission?.mode ?? draft.editor.mode;
  const role = mode === "text" ? "chat" : mode === "image" ? "image" : mode === "video" ? "video" : null;
  const top = role ? selectPriorTurnEditConfiguration(saved, { role }) : { ...saved, settings: {}, explicitSettingsKeys: [],
    presetChoice: { kind: "inherit" as const }, profileChoice: { kind: "inherit" as const },
    workflowChoice: { kind: "inherit" as const }, visionProfileChoice: { kind: "inherit" as const } };
  const inherited = role ? priorTurnEditConfigurationSource(top, { role }) : undefined;
  const request = buildLegacyPriorTurnEditRequest({ ...top, source: { ...top.source, settings: inherited?.settings ?? {} } },
    submission ? { ...submission, settings: { ...top.settings, ...draft.editor.templateSettings?.settings } } : undefined);
  const roles = Object.fromEntries(Object.entries(saved.configurations!.roles).filter(([key]) => key !== role));
  if (Object.keys(roles).length) request.role_overrides = snapshot(roles);
  if (Object.keys(saved.configurations!.steps).length) request.step_overrides = snapshot(saved.configurations!.steps);
  return request;
}

/** Fingerprint the wire payload and target, never the transient editor request ID. */
export function priorTurnEditRequestFingerprint(source: PriorTurnEditSource, request: PriorTurnEditRequest): string {
  const payload: Partial<PriorTurnEditRequest> = snapshot(request);
  delete payload.idempotency_key;
  return JSON.stringify(canonical({ chat_id: source.chat_id, message_id: source.source_user_message_id, request: payload }));
}

export function writePriorTurnEditDraft(draft: PriorTurnEditDraft, storage?: PriorTurnEditStorage): void {
  try {
    (storage ?? localStorage).setItem(key(draft.source.chat_id, draft.source.source_user_message_id), JSON.stringify(draft));
  } catch {
    throw new Error(STORAGE_ERROR);
  }
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function nonempty(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function strings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(nonempty);
}

function mode(value: unknown): boolean {
  return typeof value === "string" && ["auto", "text", "image", "video"].includes(value);
}

function count(value: unknown): boolean {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 1;
}

function scalarChoice(value: unknown): boolean {
  return record(value) && (value.kind === "inherit" || (value.kind === "explicit" && (value.value === null || nonempty(value.value))));
}

function workflowInput(value: unknown): boolean {
  if (value === null) return true;
  if (!record(value) || !["chat", "image", "video"].includes(String(value.selector_capability))) return false;
  if (value.mode === "family") return nonempty(value.workflow_family_id) && value.workflow_revision_id === undefined;
  if (value.mode === "revision") return nonempty(value.workflow_revision_id) && value.workflow_family_id === undefined;
  return (value.mode === "default" || value.mode === "automatic")
    && value.workflow_family_id === undefined && value.workflow_revision_id === undefined;
}

function workflowChoice(value: unknown): boolean {
  return record(value) && (value.kind === "inherit" || (value.kind === "explicit" && workflowInput(value.value)));
}

function configurationOverrides(value: unknown): boolean {
  return record(value) && Object.entries(value).every(([name, item]) => name === "settings" ? record(item)
    : name === "workflow_selection" ? workflowInput(item)
      : ["preset_id", "profile_id", "vision_profile_id"].includes(name) && (item === null || nonempty(item)));
}

function configurationShape(value: unknown, source: PriorTurnEditSource): boolean {
  if (!record(value) || !record(value.target) || !record(value.roles) || !record(value.steps)) return false;
  const stepIds = new Set(source.steps?.map((step) => step.step_id) ?? []);
  return (Object.keys(value.target).length === 1 && ("role" in value.target
    ? ["chat", "image", "video"].includes(String(value.target.role)) : stepIds.has(String(value.target.stepId))))
    && Object.entries(value.roles).every(([role, changes]) => ["chat", "image", "video"].includes(role) && configurationOverrides(changes))
    && Object.entries(value.steps).every(([stepId, changes]) => stepIds.has(stepId) && configurationOverrides(changes));
}

function sourceReference(value: unknown): boolean {
  return record(value) && nonempty(value.reference_subject_id) && typeof value.mention_slug === "string"
    && typeof value.subject_name === "string" && typeof value.subject_kind === "string" && typeof value.source === "string"
    && (value.role === undefined || value.role === null || typeof value.role === "string")
    && (value.strength === undefined || value.strength === null || typeof value.strength === "number")
    && strings(value.reference_asset_ids_json) && strings(value.artifact_ids_json);
}

function sourceShape(value: Record<string, unknown>, chatId: string, messageId: string): boolean {
  return value.chat_id === chatId && value.source_user_message_id === messageId && nonempty(value.source_run_id)
    && nonempty(value.source_snapshot_sha256) && record(value.settings) && record(value.resolved_settings)
    && mode(value.mode) && count(value.output_count) && typeof value.text === "string"
    && ["chat", "image", "video"].includes(String(value.settings_role)) && strings(value.input_artifact_ids)
    && Array.isArray(value.input_artifacts) && value.input_artifacts.every((item) => record(item) && nonempty(item.id) && nonempty(item.media_type))
    && Array.isArray(value.references) && value.references.every(sourceReference)
    && Array.isArray(value.context_messages) && value.context_messages.every((item) => record(item) && typeof item.role === "string" && typeof item.content === "string");
}

function editorShape(value: Record<string, unknown>): boolean {
  return nonempty(value.requestId) && mode(value.mode) && count(value.outputCount)
    && ["inherit", "replace"].includes(String(value.attachmentIntent)) && ["inherit", "replace"].includes(String(value.referenceIntent))
    && Array.isArray(value.attachments) && value.attachments.every((item) => record(item) && nonempty(item.id)
      && (item.kind === "image" || item.kind === "video") && ["uploaded", "generated", "edited"].includes(String(item.origin)))
    && Array.isArray(value.mentions) && value.mentions.every((item) => record(item) && nonempty(item.referenceSubjectId) && nonempty(item.mentionSlug))
    && (value.templateSettings === null || (record(value.templateSettings) && typeof value.templateSettings.name === "string" && record(value.templateSettings.settings)))
    && (value.submittedFingerprint === undefined || typeof value.submittedFingerprint === "string");
}

/** Malformed or inaccessible data is retained and reported, never silently replaced. */
export function readPriorTurnEditDraft(chatId: string, messageId: string, storage?: PriorTurnEditStorage): PriorTurnEditDraft | null {
  try {
    const raw = (storage ?? localStorage).getItem(key(chatId, messageId));
    if (raw === null) return null;
    const value: unknown = JSON.parse(raw);
    if (!record(value) || value.version !== 1 || !record(value.source) || !sourceShape(value.source, chatId, messageId)
      || !record(value.composer) || typeof value.composer.text !== "string" || !record(value.editor)
      || !(value.composer.promptSource === null || record(value.composer.promptSource))
      || !editorShape(value.editor) || !record(value.settings)
      || !["chat", "image", "video"].includes(String(value.settingsRole))
      || ![value.presetChoice, value.profileChoice, value.visionProfileChoice].every(scalarChoice)
      || !workflowChoice(value.workflowChoice)
      || (value.explicitSettingsKeys !== undefined && !strings(value.explicitSettingsKeys))
      || (value.removedReferenceSubjectIds !== undefined && !strings(value.removedReferenceSubjectIds))) {
      throw new Error(READ_ERROR);
    }
    const draft = value as unknown as PriorTurnEditDraft;
    if (value.configurations !== undefined && !configurationShape(value.configurations, draft.source)) throw new Error(READ_ERROR);
    if (value.pending !== undefined) {
      if (!record(value.pending) || !record(value.pending.request) || !nonempty(value.pending.request.idempotency_key)
        || (value.pending.editableFingerprint !== undefined && !nonempty(value.pending.editableFingerprint))
        || (value.pending.request.confirm_media !== undefined && typeof value.pending.request.confirm_media !== "boolean")
        || typeof value.pending.request.text !== "string" || !record(value.pending.request.settings)
        || !mode(value.pending.request.mode) || !count(value.pending.request.output_count)
        || value.pending.request.source_run_id !== draft.source.source_run_id
        || value.pending.request.source_snapshot_sha256 !== draft.source.source_snapshot_sha256
        || value.pending.fingerprint !== priorTurnEditRequestFingerprint(draft.source, draft.pending!.request)) throw new Error(READ_ERROR);
    }
    if (draft.configurations) return draft;
    const previousEditable = priorTurnEditRequestFingerprint(draft.source, buildLegacyPriorTurnEditRequest(draft));
    const next = storeConfiguration(draft);
    if (next.pending) next.pending = { ...next.pending, editableFingerprint:
      previousEditable === (next.pending.editableFingerprint ?? next.pending.fingerprint)
        ? priorTurnEditRequestFingerprint(next.source, buildPriorTurnEditRequest(next))
        : "legacy-unsent:" + previousEditable };
    return next;
  } catch {
    throw new Error(READ_ERROR);
  }
}

/** Persist the frozen request before returning it to the caller that sends it. */
export function preparePriorTurnEditSubmission(draft: PriorTurnEditDraft, submission?: TurnEditorSubmission, storage?: PriorTurnEditStorage): { draft: PriorTurnEditDraft; request: PriorTurnEditRequest } {
  const next = snapshot(draft);
  if (submission) {
    next.composer = { text: submission.text, promptSource: submission.promptSource ?? null };
    next.editor.mode = submission.mode;
    next.editor.outputCount = submission.outputCount;
    next.settings = snapshot(submission.settings);
    if (next.configurations) for (const name of Object.keys(next.editor.templateSettings?.settings ?? {})) {
      if (Object.hasOwn(draft.settings, name)) next.settings[name] = snapshot(draft.settings[name]);
      else delete next.settings[name];
    }
    next.settingsRole = submission.settingsRole;
  }
  let request = buildPriorTurnEditRequest(next, submission);
  const editableFingerprint = priorTurnEditRequestFingerprint(next.source, request);
  const previousEditable = next.pending?.editableFingerprint ?? next.pending?.fingerprint;
  if (next.pending && previousEditable === editableFingerprint) {
    request = snapshot(next.pending.request);
  } else if (next.pending) {
    request.idempotency_key = crypto.randomUUID();
  }
  next.editor.requestId = request.idempotency_key;
  next.pending = {
    editableFingerprint, fingerprint: priorTurnEditRequestFingerprint(next.source, request),
    request: snapshot(request),
  };
  writePriorTurnEditDraft(next, storage);
  return { draft: next, request };
}

/** Persist explicit confirmation before sending; retain the editable draft identity for retries. */
export function confirmPriorTurnEditSubmission(draft: PriorTurnEditDraft, selectedMode: RoutingMode, storage?: PriorTurnEditStorage): { draft: PriorTurnEditDraft; request: PriorTurnEditRequest } {
  if (!draft.pending) throw new Error("The pending edited version is unavailable.");
  const next = snapshot(draft);
  const pending = next.pending!;
  const request = { ...pending.request, mode: selectedMode, confirm_media: true };
  next.pending = {
    editableFingerprint: pending.editableFingerprint ?? pending.fingerprint,
    fingerprint: priorTurnEditRequestFingerprint(next.source, request), request: snapshot(request),
  };
  writePriorTurnEditDraft(next, storage);
  return { draft: next, request };
}

/** Acceptance must stay successful if cleanup fails; avoid clearing a newer submission. */
export function removePriorTurnEditDraft(chatId: string, messageId: string, expectedRequestId?: string, storage?: PriorTurnEditStorage): void {
  try {
    const target = storage ?? localStorage;
    if (expectedRequestId !== undefined) {
      const saved = readPriorTurnEditDraft(chatId, messageId, target);
      if (saved?.pending?.request.idempotency_key !== expectedRequestId) return;
      if (priorTurnEditRequestFingerprint(saved.source, buildPriorTurnEditRequest(saved))
        !== (saved.pending.editableFingerprint ?? saved.pending.fingerprint)) return;
    }
    target.removeItem(key(chatId, messageId));
  } catch {
    // A retained pending request replays the same accepted work.
  }
}
