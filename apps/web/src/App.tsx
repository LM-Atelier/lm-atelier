import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDown,
  ArrowUp,
  Bot,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  Cpu,
  Download,
  Film,
  Folder,
  Gauge,
  GitBranch,
  HardDrive,
  Image as ImageIcon,
  Library,
  LoaderCircle,
  Menu,
  MessageSquare,
  MoreHorizontal,
  Paperclip,
  Pencil,
  Plus,
  Quote,
  RotateCcw,
  Search,
  Send,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  Wand2,
  Workflow as WorkflowIcon,
  X,
} from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { CopyTextButton } from "./CopyTextButton";
import { CredentialSettingsCard } from "./CredentialSettingsCard";
import { InstallConfirmDialog } from "./InstallConfirmDialog";
import { api } from "./api";
import {
  supportLinks,
  downloadJson,
  formatBytes,
  formatDate,
} from "./format";
import {
  calibratedImageEditStrength,
  estimateImageEditStrength,
  IMAGE_EDIT_STRENGTH_MODE_KEY,
  resolveImageEditStrengthMode,
  type ImageEditStrengthMode,
  type WorkflowImageEditCalibration,
  workflowImageEditCalibration,
} from "./imageEditStrength";
import { GlobalNotices } from "./GlobalNotices";
import {
  artifactSource,
  mediaOriginForPart,
  mediaOriginLabel,
  editLineageForResult,
  editSourceUrlForResult,
  messagePartsForTranscript,
  priorVisibleMediaByMessage,
  type EditLineageStep,
  type MediaOrigin,
} from "./messageMedia";
import { useLiveEvents } from "./useLiveEvents";
import { DownloadDiagnosticsButton } from "./DownloadDiagnosticsButton";
import { StatusDot } from "./StatusDot";
import { ErrorCallout } from "./ErrorCallout";
import { EmptyState } from "./EmptyState";
import { AtelierMark } from "./AtelierMark";
import { EditingStudio } from "./EditingStudio";
import { MessageTimestamp } from "./MessageTimestamp";
import { PendingResponseStatus } from "./PendingResponseStatus";
import { MarkdownText } from "./MarkdownText";
import { focusMainContent, roleForMode } from "./viewHelpers";
import { ArtifactPart } from "./ArtifactPart";
import { FirstRunSetup, SetupWizard } from "./SetupWizard";
import { CustomNodesPanel } from "./CustomNodesPanel";
import { MediaLibraryView } from "./MediaLibraryView";
import { MediaOutputPlan } from "./MediaOutputPlan";
import { ModelCard } from "./ModelCard";
import { ModelUpdatesPanel } from "./ModelUpdatesPanel";
import { RuntimeSetupCard } from "./RuntimeSetupCard";
import { RegistryInstallsPanel } from "./RegistryInstallsPanel";
import { useProjectMutations } from "./useProjectMutations";
import { WorkflowPackageReview } from "./WorkflowPackageReview";
import { useWorkflowPackageImport } from "./useWorkflowPackageImport";
import { JobsPanel } from "./JobsPanel";
import { editVisionNote, workshopTranscript } from "./promptWorkshop";
import { WorkerLogFolderButton, WorkerStartupLimit } from "./WorkerStartupLimit";
import { WorkerStatusCard } from "./WorkerStatusCard";
import { useComposerUploads } from "./useComposerUploads";
import type { ComposerAttachment } from "./useComposerUploads";
import { useDraftClassification } from "./useDraftClassification";
import { useFirstRunSetup } from "./useFirstRunSetup";
import { useGenerationModeSelection } from "./useGenerationModeSelection";
import { useMessageActions } from "./useMessageActions";
import {
  normalizeSettingsForFields,
  promptPreviewSettings,
  resolveCapabilitySettings,
  resolveWorkflowSettings,
} from "./settings";
import type {
  ApplicationInfo,
  ArtifactLibraryItem,
  BackupInfo,
  CatalogModel,
  CatalogPreflight,
  Chat,
  ChatDetail,
  EngineCapabilities,
  EngineRole,
  GenerationPreset,
  GenerationPresetBundle,
  Message,
  MessagePart,
  ModelAssetInstall,
  ModelInstall,
  ModelProfile,
  ModelProfileBundle,
  Project,
  ReferenceRecipe,
  RoutingMode,
  RuntimeStatus,
  SetupReadinessReport,
  SettingField,
  SystemInfo,
  TurnAccepted,
  Workflow,
  WorkPlan,
} from "./types";

type View = "chat" | "media" | "models" | "workflows" | "settings";
type Visibility = "basic" | "advanced" | "expert";
type PendingTurn = { id: string; text: string; mode: RoutingMode };
type VisualTarget = {
  attachment: ComposerAttachment;
  // null means "attach only", used by Reference: quoting a picture says what
  // the next turn is about; it must not silently change the generation mode
  // the user chose, the way Edit and Animate deliberately do.
  mode: "image" | "video" | null;
  requestId: number;
  // Open the editing studio once the image is attached - the library's Edit
  // entry point, where no instruction has been drafted yet.
  studio?: boolean;
  // Companions from a library multi-select; the studio's "Apply to each"
  // path then runs one turn per image.
  extraAttachments?: ComposerAttachment[];
};
type SendTurnVariables = PendingTurn & {
  chatId: string;
  artifacts: string[];
  settings: Record<string, unknown>;
  stopCurrent?: boolean;
};

const visibilityRank: Record<Visibility, number> = { basic: 0, advanced: 1, expert: 2 };
const AUTO_PROFILE_ID = "__auto__";
const SETUP_DISMISSED_KEY = "lm-atelier-setup-dismissed";
const CURRENT_CHAT_KEY = "local-lm-chat";

function formatTechnicalDetails(
  application: ApplicationInfo,
  system: SystemInfo,
  engines: EngineCapabilities[],
): string {
  const devices = system.devices.length
    ? [
        "Devices:",
        ...system.devices.map((device) => {
          const facts = [
            device.kind,
            device.backend,
            device.total_memory_bytes == null ? null : formatBytes(device.total_memory_bytes),
          ].filter(Boolean);
          return `- ${device.name}${facts.length ? ` (${facts.join("; ")})` : ""}`;
        }),
      ]
    : ["Devices: None detected"];
  const runtimes = engines.length
    ? [
        "Engines:",
        ...engines.map(
          (engine) => `- ${engine.engine} ${engine.version} (${engine.roles.join(", ")})`,
        ),
      ]
    : ["Engines: None reported"];
  return [
    `LM Atelier: ${application.version}`,
    `Platform: ${system.distribution} ${system.distribution_version} (${system.platform} ${system.platform_release}; ${system.architecture})`,
    `Runtime: Python ${system.python_version}`,
    `CPU: ${system.cpu_model}`,
    ...devices,
    ...runtimes,
  ].join("\n");
}

/** The library's Edit action: attach the selection in the chat composer,
 * switch to image mode, and open the studio. */
function libraryEditTarget(artifacts: ArtifactLibraryItem[]): VisualTarget | null {
  const [first, ...rest] = artifacts.map((artifact): ComposerAttachment => ({
    id: artifact.id,
    kind: "image",
    artifact,
    // Stored uploads keep their filename; generated media has none.
    origin: artifact.original_name ? "uploaded" : "generated",
  }));
  if (!first) return null;
  return {
    attachment: first,
    mode: "image",
    requestId: Date.now(),
    studio: true,
    extraAttachments: rest,
  };
}

function PartView({
  part,
  liveText,
  markdown = false,
  origin,
  onEditImage,
  onAnimateImage,
  onReferenceMedia,
  onToggleFavorite,
  compareSourceUrl,
  lineage,
}: {
  part: MessagePart;
  liveText?: string;
  markdown?: boolean;
  origin: MediaOrigin | null;
  onEditImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onAnimateImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onReferenceMedia?: (part: MessagePart, origin: MediaOrigin) => void;
  onToggleFavorite?: (part: MessagePart) => void;
  compareSourceUrl?: string | null;
  lineage?: EditLineageStep[];
}) {
  if (part.type === "text") {
    const text = liveText || part.text || "";
    return markdown ? <MarkdownText text={text} /> : <div className="message-text">{text}</div>;
  }
  if (part.type === "image" || part.type === "video" || part.type === "attachment") {
    return <ArtifactPart part={part} origin={origin} onEditImage={onEditImage} onAnimateImage={onAnimateImage} onReferenceMedia={onReferenceMedia} onToggleFavorite={onToggleFavorite} compareSourceUrl={compareSourceUrl} lineage={lineage} />;
  }
  if (part.type === "progress") {
    const progress = Number(part.metadata_json.progress ?? 0);
    const indeterminate = part.metadata_json.indeterminate === true;
    return (
      <div className="generation-progress" role="status" aria-live="polite">
        <Sparkles size={17} />
        <div>
          <span>{part.text || "Working"}</span>
          <div className="progress-track">
            <div
              className={indeterminate ? "indeterminate" : undefined}
              style={indeterminate ? undefined : { width: `${progress * 100}%` }}
            />
          </div>
        </div>
      </div>
    );
  }
  if (part.type === "error") return <div className="message-error" role="alert">{part.text}</div>;
  return <div className="message-error" role="alert">Unsupported message part: {String(part.type)}</div>;
}

function MessageBubble({
  message,
  liveText,
  hiddenInputArtifactIds,
  onRegenerate,
  onEdit,
  onSelectRevision,
  onCancelQueued,
  onEditImage,
  onAnimateImage,
  onReferenceMedia,
  onToggleFavorite,
  onQuote,
  onDeleteExchange,
  onForkThread,
  compareSourceUrl,
  lineage,
}: {
  message: Message;
  liveText?: string;
  hiddenInputArtifactIds?: ReadonlySet<string>;
  onRegenerate?: (messageId: string) => void;
  onEdit?: (messageId: string, text: string) => void;
  onSelectRevision?: (messageId: string, revisionId: string) => void;
  onCancelQueued?: () => void;
  onEditImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onAnimateImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onReferenceMedia?: (part: MessagePart, origin: MediaOrigin) => void;
  onToggleFavorite?: (part: MessagePart) => void;
  onQuote?: (text: string) => void;
  onDeleteExchange?: (messageId: string) => void;
  onForkThread?: (messageId: string) => void;
  compareSourceUrl?: string | null;
  lineage?: EditLineageStep[];
}) {
  const visibleParts = messagePartsForTranscript(message, hiddenInputArtifactIds);
  const userText = visibleParts.filter((part) => part.type === "text").map((part) => part.text || "").join("\n");
  const copyableText = (liveText || userText).trim();
  const chatProgress = visibleParts.find(
    (part) => part.type === "progress" && part.metadata_json.activity === "chat",
  );
  const hasVisibleText = Boolean(copyableText);
  const hasMediaProgress = visibleParts.some(
    (part) => part.type === "progress" && part.metadata_json.activity !== "chat",
  );
  const showChatStartup = message.role === "assistant"
    && message.status === "pending"
    && !hasVisibleText
    && (Boolean(chatProgress) || !hasMediaProgress);
  const renderedParts = chatProgress
    ? visibleParts.filter((part) => part.id !== chatProgress.id)
    : visibleParts;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(userText);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const metadata = message.parts.find((part) => part.type === "generation_metadata")?.metadata_json;
  const context = metadata?.context as Record<string, unknown> | undefined;
  const provenance = metadata?.provenance as Record<string, unknown> | undefined;
  const routing = provenance?.routing as Record<string, unknown> | undefined;
  const operation = typeof routing?.operation === "string" ? routing.operation : undefined;
  const modelSelection = provenance?.model_selection as Record<string, unknown> | undefined;
  const autoProfileName = modelSelection?.mode === "auto"
    ? String(modelSelection.profile_name ?? "")
    : "";
  const autoMatchedTerms = modelSelection?.mode === "auto" && Array.isArray(modelSelection.matched_terms)
    ? modelSelection.matched_terms.filter((term): term is string => typeof term === "string").slice(0, 3)
    : [];
  const autoSelectionDetail = autoMatchedTerms.length
    ? ` · matched ${autoMatchedTerms.join(", ")}`
    : modelSelection?.mode === "auto" && modelSelection.fallback
      ? " · general fallback"
      : "";
  const auxiliaryAssets = provenance?.auxiliary_assets as Record<string, unknown> | undefined;
  const loraSelection = auxiliaryAssets?.selection as Record<string, unknown> | undefined;
  const automaticLoras = loraSelection?.mode === "automatic" && Array.isArray(loraSelection.selected)
    ? loraSelection.selected.filter(
        (item): item is Record<string, unknown> => Boolean(item) && typeof item === "object",
      )
    : [];
  const automaticLoraNames = automaticLoras
    .map((item) => String(item.name ?? ""))
    .filter(Boolean);
  const automaticLoraTerms = Array.from(new Set(automaticLoras.flatMap((item) => (
    Array.isArray(item.matched_terms)
      ? item.matched_terms.filter((term): term is string => typeof term === "string")
      : []
  )))).slice(0, 3);
  const appliedTriggerWords = Array.isArray(auxiliaryAssets?.trigger_words_applied)
    ? auxiliaryAssets.trigger_words_applied.filter(
        (word): word is string => typeof word === "string" && Boolean(word),
      )
    : [];
  const usage = context?.usage as Record<string, unknown> | undefined;
  const inputTokens = Number(usage?.prompt_tokens ?? context?.input_tokens ?? 0);
  const contextLimit = Number(context?.context_limit ?? 0);
  const omitted = Number(context?.messages_omitted ?? 0);
  const contextCompaction = context?.compaction as Record<string, unknown> | undefined;
  const compactedMessages = contextCompaction?.active
    ? Number(contextCompaction.source_message_count ?? omitted)
    : 0;
  const completedRevisions = (message.response_revisions ?? [])
    .filter((revision) => revision.status === "complete")
    .sort((left, right) => left.sequence - right.sequence);
  const activeRevisionIndex = completedRevisions.findIndex(
    (revision) => revision.id === message.active_response_revision_id,
  );
  const revisionIndex = activeRevisionIndex >= 0
    ? activeRevisionIndex
    : Math.max(0, completedRevisions.length - 1);
  const regenerationPending = (message.response_revisions ?? []).some(
    (revision) => revision.status === "pending",
  );
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "You" : <Bot size={19} />}</div>
      <div className="message-content">
        {editing ? <div className="message-edit"><textarea aria-label="Edit message" rows={4} value={draft} onChange={(event) => setDraft(event.target.value)} /><div><button onClick={() => { setDraft(userText); setEditing(false); }}>Cancel</button><button className="primary" disabled={!draft.trim()} onClick={() => { onEdit?.(message.id, draft.trim()); setEditing(false); }}>Send edited message</button></div></div> : renderedParts.map((part) => <PartView key={part.id} part={part} liveText={liveText} markdown={message.role === "assistant"} origin={mediaOriginForPart(part, operation, message.role === "assistant" ? "generated" : null)} onEditImage={onEditImage} onAnimateImage={onAnimateImage} onReferenceMedia={onReferenceMedia} onToggleFavorite={onToggleFavorite} compareSourceUrl={message.role === "assistant" ? compareSourceUrl : undefined} lineage={message.role === "assistant" ? lineage : undefined} />)}
        {liveText && !visibleParts.some((part) => part.type === "text") && (
          <MarkdownText text={liveText} />
        )}
        {showChatStartup && <PendingResponseStatus label={chatProgress?.text || "Starting chat"} startedAt={message.created_at} />}
        {message.role === "user" && message.status === "complete" && !editing && <div className="message-meta"><MessageTimestamp at={message.created_at} />{confirmingDelete ? <span className="delete-confirm"><span>Also deletes the answer and its media.</span><button className="danger" onClick={() => { setConfirmingDelete(false); onDeleteExchange?.(message.id); }}>Delete turn</button><button onClick={() => setConfirmingDelete(false)}>Keep</button></span> : <span className="message-actions">{onEdit && <button onClick={() => setEditing(true)} aria-label="Edit message" title="Edit"><Pencil size={14} /></button>}{copyableText && <CopyTextButton text={copyableText} label="Copy user message" buttonText="" />}{onDeleteExchange && <button aria-label="Delete this turn" title="Delete turn" onClick={() => setConfirmingDelete(true)}><Trash2 size={14} /></button>}</span>}</div>}
        {message.role === "assistant" && message.status === "cancelled" && !visibleParts.some((part) => part.type === "error") && (
          <div className="message-meta"><span>Generation cancelled</span></div>
        )}
        {message.role === "assistant" && message.status === "complete" && (
          <div className="message-meta">
            <MessageTimestamp at={message.created_at} />
            {autoProfileName && <span>Auto chose {autoProfileName}{autoSelectionDetail}</span>}
            {automaticLoraNames.length > 0 && (
              <span>
                LoRA Auto used {automaticLoraNames.join(", ")}
                {automaticLoraTerms.length > 0 ? ` — matched ${automaticLoraTerms.join(", ")}` : ""}
              </span>
            )}
            {appliedTriggerWords.length > 0 && <span>Added trigger words: {appliedTriggerWords.join(", ")}</span>}
            {contextLimit > 0 && (
              <span>
                Context {inputTokens.toLocaleString()} / {contextLimit.toLocaleString()} tokens
                {omitted > 0 && compactedMessages === 0
                  ? ` · ${omitted} earlier message${omitted === 1 ? "" : "s"} omitted`
                  : ""}
              </span>
            )}
            {compactedMessages > 0 && (
              <span>
                Compacted {compactedMessages} earlier message
                {compactedMessages === 1 ? "" : "s"} · full transcript preserved
              </span>
            )}
            {regenerationPending && <span>Regenerating…</span>}
            {/* Always visible, unlike the hover actions below: cycling between
                answers is navigation, and a control the user cannot see is a
                control they do not know exists. */}
            {completedRevisions.length > 1 && onSelectRevision && (
              <span className="response-revision-controls">
                <button
                  disabled={revisionIndex <= 0}
                  title="Previous answer"
                  onClick={() => onSelectRevision(
                    message.id,
                    completedRevisions[revisionIndex - 1]!.id,
                  )}
                  aria-label="Previous response revision"
                >
                  <ChevronLeft size={14} />
                </button>
                <span>{revisionIndex + 1} / {completedRevisions.length}</span>
                <button
                  disabled={revisionIndex >= completedRevisions.length - 1}
                  title="Next answer"
                  onClick={() => onSelectRevision(
                    message.id,
                    completedRevisions[revisionIndex + 1]!.id,
                  )}
                  aria-label="Next response revision"
                >
                  <ChevronRight size={14} />
                </button>
              </span>
            )}
            {/* Revealed on hover, and on keyboard focus - hover alone would
                put these actions out of reach without a mouse. */}
            <span className="message-actions">
              {copyableText && (
                <CopyTextButton
                  text={copyableText}
                  label="Copy assistant message"
                  buttonText=""
                />
              )}
              {copyableText && onQuote && (
                <button onClick={() => onQuote(copyableText)} aria-label="Quote response" title="Quote">
                  <Quote size={14} />
                </button>
              )}
              {onRegenerate && (
                <button onClick={() => onRegenerate(message.id)} aria-label="Regenerate response" title="Regenerate">
                  <RotateCcw size={14} />
                </button>
              )}
              {onForkThread && (
                <button onClick={() => onForkThread(message.id)} aria-label="Start a new thread here" title="New thread from here">
                  <GitBranch size={14} />
                </button>
              )}
            </span>
          </div>
        )}
        {message.role === "assistant" && message.status !== "complete" && copyableText && (
          <div className="message-meta"><CopyTextButton text={copyableText} label="Copy assistant message" /></div>
        )}
        {message.role === "assistant" && message.status === "pending" && onCancelQueued && (
          <div className="message-meta">
            <button onClick={onCancelQueued}><X size={13} /> Cancel queued item</button>
          </div>
        )}
      </div>
    </article>
  );
}

type LoraSetting = {
  asset_id: string;
  model_strength: number;
  clip_strength: number;
  enabled: boolean;
};

function LoraStackControl({
  value,
  onChange,
}: {
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const assets = useQuery({
    queryKey: ["model-assets", "lora"],
    queryFn: () => api.modelAssets("lora"),
  });
  const stack: LoraSetting[] = Array.isArray(value)
    ? value.map((item) => (
      item && typeof item === "object" && typeof (item as Record<string, unknown>).asset_id === "string"
        ? {
            asset_id: String((item as Record<string, unknown>).asset_id),
            model_strength: Number((item as Record<string, unknown>).model_strength ?? 1),
            clip_strength: Number((item as Record<string, unknown>).clip_strength ?? 1),
            enabled: (item as Record<string, unknown>).enabled !== false,
          }
        : {
            asset_id: "",
            model_strength: 1,
            clip_strength: 1,
            enabled: false,
          }
    ))
    : [];
  const installed = assets.data?.filter((asset) => asset.active && asset.verified_at) ?? [];
  const used = new Set(stack.map((item) => item.asset_id));
  const addable = installed.find((asset) => !used.has(asset.id));
  const update = (index: number, patch: Partial<LoraSetting>) => {
    onChange(stack.map((item, candidate) => candidate === index ? { ...item, ...patch } : item));
  };
  const move = (index: number, offset: number) => {
    const next = [...stack];
    const target = index + offset;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target], next[index]];
    onChange(next);
  };
  return (
    <div className="setting-row lora-stack-control">
      <span><strong>LoRAs</strong></span>
      <div className="lora-stack">
        {stack.map((item, index) => {
          const asset = installed.find((candidate) => candidate.id === item.asset_id);
          const metadata = asset?.manifest_json.metadata;
          const triggerWords = metadata && typeof metadata === "object"
            && Array.isArray((metadata as Record<string, unknown>).trigger_words)
            ? (metadata as Record<string, unknown>).trigger_words as string[]
            : [];
          return (
            <div className={`lora-stack-item${asset ? "" : " unavailable"}`} key={`${item.asset_id}:${index}`}>
              <select
                aria-label={`LoRA ${index + 1}`}
                value={item.asset_id}
                onChange={(event) => update(index, { asset_id: event.target.value })}
              >
                {!asset && <option value={item.asset_id}>{item.asset_id ? "Unavailable LoRA" : "Choose LoRA"}</option>}
                {installed.map((candidate) => (
                  <option value={candidate.id} key={candidate.id}>{candidate.name}</option>
                ))}
              </select>
              <label>Model<input aria-label={`LoRA ${index + 1} model strength`} type="number" min="-4" max="4" step="0.05" value={item.model_strength} onChange={(event) => update(index, { model_strength: Number(event.target.value) })} /></label>
              <label>CLIP<input aria-label={`LoRA ${index + 1} CLIP strength`} type="number" min="-4" max="4" step="0.05" value={item.clip_strength} onChange={(event) => update(index, { clip_strength: Number(event.target.value) })} /></label>
              <label className="lora-enabled"><input aria-label={`Enable LoRA ${index + 1}`} type="checkbox" checked={item.enabled} onChange={(event) => update(index, { enabled: event.target.checked })} />On</label>
              <span className="row-actions">
                <button type="button" className="secondary compact-button" aria-label={`Move LoRA ${index + 1} up`} disabled={index === 0} onClick={() => move(index, -1)}><ArrowUp size={13} /></button>
                <button type="button" className="secondary compact-button" aria-label={`Move LoRA ${index + 1} down`} disabled={index === stack.length - 1} onClick={() => move(index, 1)}><ArrowDown size={13} /></button>
                <button type="button" className="secondary compact-button danger" aria-label={`Remove LoRA ${index + 1}`} onClick={() => onChange(stack.filter((_, candidate) => candidate !== index))}><X size={13} /></button>
              </span>
              {(asset?.family || triggerWords.length > 0) && <small>{[asset?.family, ...triggerWords.slice(0, 3)].filter(Boolean).join(" · ")}</small>}
            </div>
          );
        })}
        <button
          type="button"
          className="secondary compact-button"
          disabled={!addable || stack.length >= 8}
          onClick={() => addable && onChange([
            ...stack,
            {
              asset_id: addable.id,
              model_strength: 1,
              clip_strength: 1,
              enabled: true,
            },
          ])}
        >
          <Plus size={13} /> Add LoRA
        </button>
      </div>
    </div>
  );
}

function SettingControl({
  field,
  value,
  onChange,
}: {
  field: SettingField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const fixed = field.choices.length === 1;
  if (field.key === "loras") {
    return <LoraStackControl value={value} onChange={onChange} />;
  }
  if (field.type === "boolean") {
    return (
      <label className="setting-row toggle-row" title={field.help || undefined}>
        <span><strong>{field.label}</strong>{fixed && field.help && <small>{field.help}</small>}</span>
        <input type="checkbox" checked={Boolean(value)} disabled={fixed} onChange={(event) => onChange(event.target.checked)} />
      </label>
    );
  }
  if (field.type === "enum") {
    return (
      <label className="setting-row" title={field.help || undefined}>
        <span><strong>{field.label}</strong>{fixed && field.help && <small>{field.help}</small>}</span>
        <select value={String(value ?? "")} disabled={fixed} onChange={(event) => onChange(event.target.value)}>
          {field.choices.map((choice) => <option key={String(choice)}>{String(choice)}</option>)}
        </select>
      </label>
    );
  }
  if (field.type === "number" || field.type === "integer") {
    return (
      <label className="setting-row" title={field.help || undefined}>
        <span><strong>{field.label}</strong>{fixed && field.help && <small>{field.help}</small>}</span>
        <input
          type="number"
          value={Number(value ?? field.default)}
          min={field.minimum ?? undefined}
          max={field.maximum ?? undefined}
          step={field.step ?? (field.type === "integer" ? 1 : 0.01)}
          disabled={fixed}
          onChange={(event) => onChange(field.type === "integer" ? Number.parseInt(event.target.value) : Number(event.target.value))}
        />
      </label>
    );
  }
  if (field.type === "array" || field.type === "object") {
    return (
      <label className="setting-row" title={field.help || undefined}>
        <span><strong>{field.label}</strong>{fixed && field.help && <small>{field.help}</small>}</span>
        <textarea
          rows={3}
          disabled={fixed}
          defaultValue={JSON.stringify(value ?? field.default, null, 2)}
          onBlur={(event) => {
            try {
              const parsed = JSON.parse(event.target.value) as unknown;
              event.target.setCustomValidity("");
              onChange(parsed);
            } catch {
              event.target.setCustomValidity("Enter valid JSON");
              event.target.reportValidity();
            }
          }}
        />
      </label>
    );
  }
  return (
    <label className="setting-row" title={field.help || undefined}>
      <span><strong>{field.label}</strong>{fixed && field.help && <small>{field.help}</small>}</span>
      <input value={String(value ?? "")} disabled={fixed} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function ImageEditStrengthControl({
  field,
  parameter,
  calibration,
  resolvedSteps,
  prompt,
  layers,
  numericManualLayers,
  values,
  onValues,
}: {
  field: SettingField;
  parameter: string;
  calibration: WorkflowImageEditCalibration | null;
  resolvedSteps: unknown;
  prompt: string;
  layers: Array<Record<string, unknown> | undefined>;
  numericManualLayers: boolean[];
  values: Record<string, unknown>;
  onValues: (values: Record<string, unknown>) => void;
}) {
  const mode: ImageEditStrengthMode = resolveImageEditStrengthMode(
    parameter,
    layers,
    numericManualLayers,
  );
  const activeCalibration = calibration ? {
    ...calibration,
    minimum: field.minimum ?? calibration.minimum,
    maximum: field.maximum ?? calibration.maximum,
  } : null;
  const estimate = activeCalibration
    ? calibratedImageEditStrength(prompt, activeCalibration, resolvedSteps)
    : estimateImageEditStrength(prompt, field.minimum ?? 0, field.maximum ?? 1);
  let manualValue = estimate.value;
  for (const layer of layers) {
    if (typeof layer?.[parameter] === "number") manualValue = layer[parameter];
  }
  const selectAuto = () => {
    const next: Record<string, unknown> = {
      ...values,
      [IMAGE_EDIT_STRENGTH_MODE_KEY]: "auto",
    };
    delete next[parameter];
    onValues(next);
  };
  const selectManual = () => onValues({
    ...values,
    [IMAGE_EDIT_STRENGTH_MODE_KEY]: "manual",
    [parameter]: typeof values[parameter] === "number" ? values[parameter] : manualValue,
  });
  return (
    <div className="setting-row image-edit-strength-control">
      <span>
        <strong>Change strength</strong>
        <small>{mode === "auto" ? `Predicted: ${estimate.scope}` : "Set for this chat"}</small>
      </span>
      <div className="image-edit-strength-inputs">
        <div className="segmented compact" role="group" aria-label="Image edit change strength mode">
          <button type="button" aria-pressed={mode === "auto"} className={mode === "auto" ? "active" : ""} onClick={selectAuto}>Auto</button>
          <button type="button" aria-pressed={mode === "manual"} className={mode === "manual" ? "active" : ""} onClick={selectManual}>Manual</button>
        </div>
        {mode === "manual" && (
          <input
            aria-label="Manual change strength"
            type="number"
            value={manualValue}
            min={field.minimum ?? undefined}
            max={field.maximum ?? undefined}
            step={field.step ?? 0.01}
            onChange={(event) => onValues({
              ...values,
              [IMAGE_EDIT_STRENGTH_MODE_KEY]: "manual",
              [parameter]: Number(event.target.value),
            })}
          />
        )}
      </div>
    </div>
  );
}
function GenerationSettingsPanel({
  role,
  engines,
  values,
  onValues,
  presets,
  presetId,
  onPreset,
  workflowSchema,
  inheritedValues = {},
  inheritedPresetId = null,
  profileValues = {},
  imageEdit = false,
  imageEditPrompt = "",
  presetLabel = `${role} preset`,
  resetLabel,
  onReset,
}: {
  role: EngineRole;
  engines: EngineCapabilities[];
  values: Record<string, unknown>;
  onValues: (values: Record<string, unknown>) => void;
  presets: GenerationPreset[];
  presetId: string | null;
  onPreset: (presetId: string | null) => void;
  workflowSchema?: Record<string, unknown>;
  inheritedValues?: Record<string, unknown>;
  inheritedPresetId?: string | null;
  profileValues?: Record<string, unknown>;
  imageEdit?: boolean;
  imageEditPrompt?: string;
  presetLabel?: string;
  resetLabel: string;
  onReset: () => void;
}) {
  const [visibility, setVisibility] = useState<Visibility>("basic");
  const engine = engines.find((item) => item.roles.includes(role));
  const rolePresets = presets.filter((preset) => preset.role === role);
  const defaultPreset = rolePresets.find((preset) => preset.is_default);
  const inheritedPreset = rolePresets.find((preset) => preset.id === inheritedPresetId);
  const selectedPreset = rolePresets.find((preset) => preset.id === presetId);
  const inheritedName = inheritedPreset?.name ?? defaultPreset?.name;
  const allFields = resolveWorkflowSettings(
    resolveCapabilitySettings(engine, role),
    workflowSchema,
  );
  const editCalibration = workflowImageEditCalibration(workflowSchema);
  const strengthParameter = editCalibration?.parameter ?? "denoise";
  const strengthField = allFields.find(
    (field) => field.key === strengthParameter && field.available,
  );
  const visibleFields = allFields.filter(
    (field) =>
      field.scope !== "load"
      && visibilityRank[field.visibility] <= visibilityRank[visibility]
      && field.available
      && field.key !== strengthParameter,
  );
  // LoRAs are a list of assets with their own strengths, not one more number
  // among steps and guidance. They get their own section so choosing one is a
  // deliberate act rather than scrolling past it.
  const loraField = visibleFields.find((field) => field.key === "loras");
  const fields = visibleFields.filter((field) => field.key !== "loras");
  const effectiveValue = (field: SettingField): unknown => {
    let value = field.default;
    for (const layer of [
      profileValues,
      defaultPreset?.settings_json,
      inheritedPreset?.settings_json,
      inheritedValues,
      selectedPreset?.settings_json,
      values,
    ]) {
      if (layer && Object.prototype.hasOwnProperty.call(layer, field.key)) {
        value = layer[field.key];
      }
    }
    return value;
  };
  const stepsField = editCalibration?.stepsParameter
    ? allFields.find((field) => field.key === editCalibration.stepsParameter)
    : undefined;
  const resolvedEditSteps = stepsField ? effectiveValue(stepsField) : undefined;
  return (
    <div className="generation-settings-panel">
      <div className="segmented compact">
        {(["basic", "advanced", "expert"] as Visibility[]).map((level) => (
          <button
            key={level}
            type="button"
            className={visibility === level ? "active" : ""}
            aria-pressed={visibility === level}
            onClick={() => setVisibility(level)}
          >
            {level}
          </button>
        ))}
      </div>
      <div className="settings-list">
        <label className="setting-row">
          <span><strong>Preset</strong></span>
          <select
            aria-label={presetLabel}
            value={presetId ?? ""}
            onChange={(event) => onPreset(event.target.value || null)}
          >
            <option value="">{inheritedName ? `Inherit · ${inheritedName}` : "Inherit default"}</option>
            {rolePresets.map((preset) => (
              <option key={preset.id} value={preset.id}>{preset.name}</option>
            ))}
          </select>
        </label>
        {imageEdit && strengthField && (
          <ImageEditStrengthControl
            field={strengthField}
            parameter={strengthParameter}
            calibration={editCalibration}
            resolvedSteps={resolvedEditSteps}
            prompt={imageEditPrompt}
            layers={[
              profileValues,
              defaultPreset?.settings_json,
              inheritedPreset?.settings_json,
              inheritedValues,
              selectedPreset?.settings_json,
              values,
            ]}
            numericManualLayers={[false, false, true, true, true, true]}
            values={values}
            onValues={onValues}
          />
        )}
        {fields.map((field) => (
          <SettingControl
            key={`${field.scope}:${field.key}:${JSON.stringify(values[field.key])}`}
            field={field}
            value={effectiveValue(field)}
            onChange={(value) => onValues({ ...values, [field.key]: value })}
          />
        ))}
        {!engine && <p className="muted">No {role} engine is configured.</p>}
      </div>
      {loraField && (
        <section className="settings-section" aria-label="LoRAs">
          <h4>LoRAs</h4>
          <div className="settings-list">
            <SettingControl
              field={loraField}
              value={effectiveValue(loraField)}
              onChange={(value) => onValues({ ...values, [loraField.key]: value })}
            />
          </div>
        </section>
      )}
      <div className="generation-settings-actions">
        <button className="secondary" type="button" onClick={onReset}>{resetLabel}</button>
      </div>
    </div>
  );
}

function SettingsDrawer({
  open,
  onClose,
  mode,
  engines,
  values,
  onValues,
  presets,
  presetId,
  onPreset,
  workflowSchema,
  inheritedValues,
  inheritedPresetId,
  profileValues,
  imageEdit,
  imageEditPrompt,
}: {
  open: boolean;
  onClose: () => void;
  mode: RoutingMode;
  engines: EngineCapabilities[];
  values: Record<string, unknown>;
  onValues: (values: Record<string, unknown>) => void;
  presets: GenerationPreset[];
  presetId: string | null;
  onPreset: (presetId: string | null) => void;
  workflowSchema?: Record<string, unknown>;
  inheritedValues?: Record<string, unknown>;
  inheritedPresetId?: string | null;
  profileValues?: Record<string, unknown>;
  imageEdit: boolean;
  imageEditPrompt: string;
}) {
  const role = roleForMode(mode);
  if (!open) return null;
  return (
    <AccessibleDialog
      title={`${role[0].toUpperCase() + role.slice(1)} settings`}
      eyebrow="Chat defaults"
      closeLabel="Close settings"
      onClose={onClose}
      className="settings-drawer"
      backdropClassName="settings-drawer-backdrop"
    >
      <GenerationSettingsPanel
        role={role}
        engines={engines}
        values={values}
        onValues={onValues}
        presets={presets}
        presetId={presetId}
        onPreset={onPreset}
        workflowSchema={workflowSchema}
        inheritedValues={inheritedValues}
        inheritedPresetId={inheritedPresetId}
        profileValues={profileValues}
        imageEdit={imageEdit}
        imageEditPrompt={imageEditPrompt}
        resetLabel="Reset chat overrides"
        onReset={() => onValues({})}
      />
    </AccessibleDialog>
  );
}

function promptHelperMessageText(message: Message): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("\n")
    .trim();
}

function PromptHelperDialog({
  sourceChat,
  initialDraft,
  engines,
  workflows,
  editSourceArtifactIds,
  onAccept,
  onClose,
}: {
  sourceChat: ChatDetail;
  initialDraft: string;
  engines: EngineCapabilities[];
  workflows: Workflow[];
  // When improving an image-edit instruction, the source image rides along so
  // a vision-capable helper grounds its rewrite in what the picture shows.
  editSourceArtifactIds?: string[];
  onAccept: (draft: string) => void;
  onClose: () => void;
}) {
  const client = useQueryClient();
  const started = useRef(false);
  const [adoptedAssistantId, setAdoptedAssistantId] = useState<string | null>(null);
  const [helperId, setHelperId] = useState<string | null>(null);
  const [draft, setDraft] = useState(initialDraft);
  const [instruction, setInstruction] = useState("");
  const [working, setWorking] = useState(true);
  const [closing, setClosing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const helper = useQuery({
    queryKey: ["prompt-helper", helperId],
    queryFn: () => api.promptHelper(helperId!),
    enabled: Boolean(helperId),
    refetchInterval: 500,
  });

  const refresh = useCallback((id: string) => {
    void client.invalidateQueries({ queryKey: ["prompt-helper", id] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
  }, [client]);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    let active = true;
    void (async () => {
      try {
        const created = await api.createPromptHelper(sourceChat.id, initialDraft);
        if (!active) {
          await api.deletePromptHelper(created.id).catch(() => undefined);
          return;
        }
        setHelperId(created.id);
        await api.sendTurn(
          created.id,
          editSourceArtifactIds?.length
            ? "Improve the current draft as an editing instruction for the attached source image. "
              + "Ground it in what the image actually shows and preserve everything it does not ask to change. "
              + "Return the complete revised prompt only."
            : "Improve the current draft. Return the complete revised prompt only.",
          "text",
          editSourceArtifactIds ?? [],
          {},
        );
        if (active) refresh(created.id);
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "Could not start prompt workshop");
      } finally {
        if (active) setWorking(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [editSourceArtifactIds, initialDraft, refresh, sourceChat.id]);

  const helperMessages = helper.data ? activeBranchMessages(helper.data) : [];
  const transcript = workshopTranscript(helperMessages);
  const pending = helperMessages.some((message) => message.status === "pending");
  const latestAssistant = [...helperMessages].reverse().find(
    (message) => message.role === "assistant" && message.status === "complete",
  );
  const latestAssistantText = latestAssistant ? promptHelperMessageText(latestAssistant) : "";
  const visionNote = editVisionNote(latestAssistant, Boolean(editSourceArtifactIds?.length));

  const send = async (mode: "text" | "image" | "video", text: string) => {
    if (!helperId || !draft.trim() || !text.trim()) return;
    setWorking(true);
    setError(null);
    try {
      await api.updatePromptHelper(helperId, draft.trim());
      const role = roleForMode(mode);
      const engine = engines.find((item) => item.roles.includes(role));
      const schema = workflowSchemaForTurn(workflows, undefined, mode, false);
      const fields = resolveWorkflowSettings(resolveCapabilitySettings(engine, role), schema);
      await api.sendTurn(
        helperId,
        text.trim(),
        mode,
        [],
        mode === "text" ? {} : promptPreviewSettings(fields),
      );
      setInstruction("");
      refresh(helperId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prompt workshop request failed");
    } finally {
      setWorking(false);
    }
  };

  const finish = async (accept: boolean) => {
    if (accept && !draft.trim()) return;
    setClosing(true);
    setError(null);
    try {
      if (helperId) await api.deletePromptHelper(helperId);
      if (accept) onAccept(draft.trim());
      else onClose();
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not close prompt workshop");
      setClosing(false);
    }
  };

  const unavailable = working || pending || closing || !helperId;
  return (
    <AccessibleDialog
      title="Prompt workshop"
      eyebrow="Draft with a local model"
      closeLabel="Cancel prompt workshop"
      onClose={() => void finish(false)}
      className="prompt-helper-dialog"
    >
      <div className="prompt-helper-body">
        <label className="prompt-helper-draft">
          <span>Draft prompt</span>
          <textarea
            aria-label="Draft prompt"
            value={draft}
            maxLength={20_000}
            rows={7}
            onChange={(event) => setDraft(event.target.value)}
          />
        </label>
        <div className="prompt-helper-conversation" aria-live="polite">
          {transcript.length === 0 && !error && (
            <div className="submission-progress"><LoaderCircle size={17} /><span>Starting workshop…</span></div>
          )}
          {transcript.map((message) => <MessageBubble key={message.id} message={message} />)}
        </div>
        {visionNote && <small className="prompt-helper-vision-note">{visionNote}</small>}
        {latestAssistant && latestAssistantText && adoptedAssistantId !== latestAssistant.id && (
          <button
            type="button"
            className="secondary compact-button prompt-helper-adopt"
            disabled={unavailable}
            onClick={() => {
              setAdoptedAssistantId(latestAssistant.id);
              setDraft(latestAssistantText);
              if (helperId) {
                void api.updatePromptHelper(helperId, latestAssistantText).catch(() => undefined);
              }
            }}
          >
            <Check size={14} /> Use latest response as draft
          </button>
        )}
        <div className="prompt-helper-compose">
          <textarea
            aria-label="Prompt workshop instruction"
            placeholder="Ask for a change…"
            value={instruction}
            rows={2}
            onChange={(event) => setInstruction(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send("text", instruction);
              }
            }}
          />
          <button
            type="button"
            className="primary compact-button"
            disabled={unavailable || !instruction.trim()}
            onClick={() => void send("text", instruction)}
          >
            <Send size={14} /> Ask helper
          </button>
        </div>
        <div className="prompt-helper-actions">
          <span>
            <button
              type="button"
              className="secondary compact-button"
              disabled={unavailable || !engines.some((engine) => engine.roles.includes("image"))}
              onClick={() => void send("image", draft)}
            >
              <ImageIcon size={14} /> Preview image
            </button>
            <button
              type="button"
              className="secondary compact-button"
              disabled={unavailable || !engines.some((engine) => engine.roles.includes("video"))}
              onClick={() => void send("video", draft)}
            >
              <Film size={14} /> Preview video
            </button>
          </span>
          <span>
            <button type="button" disabled={closing} onClick={() => void finish(false)}>Cancel</button>
            <button
              type="button"
              className="primary"
              disabled={unavailable || !draft.trim()}
              onClick={() => void finish(true)}
            >
              <Check size={15} /> Use prompt
            </button>
          </span>
        </div>
        {(error || helper.error) && (
          <div className="callout error" role="alert">{error || helper.error?.message}</div>
        )}
      </div>
    </AccessibleDialog>
  );
}
function Composer({
  chat,
  engines,
  profiles,
  stoppable,
  settings,
  onSettings,
  presets,
  presetId,
  onPreset,
  onMode,
  onSend,
  onStop,
  onStopAndSend,
  workflows,
  project,
  visualTarget,
  quoteTarget,
}: {
  chat: ChatDetail;
  engines: EngineCapabilities[];
  profiles: ModelProfile[];
  stoppable: boolean;
  settings: Record<string, unknown>;
  onSettings: (settings: Record<string, unknown>) => void;
  presets: GenerationPreset[];
  presetId: string | null;
  onPreset: (presetId: string | null) => void;
  onMode: (mode: RoutingMode) => void;
  onSend: (text: string, mode: RoutingMode, artifacts: string[], settings: Record<string, unknown>) => void;
  onStop: () => void;
  onStopAndSend: (
    text: string,
    mode: RoutingMode,
    artifacts: string[],
    settings: Record<string, unknown>,
  ) => void;
  workflows: Workflow[];
  project?: Project;
  visualTarget?: VisualTarget | null;
  quoteTarget?: { text: string; requestId: number } | null;
}) {
  const [text, setText] = useState("");
  const { mode, changeMode, currentMode } = useGenerationModeSelection(chat.routing_mode, onMode);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [promptHelperDraft, setPromptHelperDraft] = useState<string | null>(null);
  const [studioOpen, setStudioOpen] = useState(false);
  const [templateSettings, setTemplateSettings] = useState<{ name: string; settings: Record<string, unknown> } | null>(null);
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  const { uploading, uploadError, setUploadError, uploadFiles } = useComposerUploads(
    (attachment) => setAttachments((current) => [...current, attachment]),
  );
  const [dropActive, setDropActive] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const textInput = useRef<HTMLTextAreaElement>(null);
  const consumedVisualRequest = useRef<number | null>(null);
  useEffect(() => {
    if (!visualTarget || consumedVisualRequest.current === visualTarget.requestId) return;
    consumedVisualRequest.current = visualTarget.requestId;
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
  }, [visualTarget, changeMode]);
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
  }, [quoteTarget]);
  const branchMessages = activeBranchMessages(chat);
  const priorVisual = branchMessages.some((message) =>
    message.parts.some((part) =>
      Boolean(part.artifact_id)
      && (part.type === "image" || part.type === "video")
      && part.metadata_json.preview !== true
    )
  );
  const priorImage = branchMessages.some((message) =>
    message.parts.some((part) =>
      Boolean(part.artifact_id)
      && part.type === "image"
      && part.metadata_json.preview !== true
    )
  );
  const usePriorVisual = useDraftClassification(chat.id, text, mode, priorVisual);
  const imageEdit = mode === "image" && (
    attachments.some((attachment) => attachment.kind === "image")
    || (priorImage && usePriorVisual)
  );
  const imageProfile = profiles.find((profile) => profile.id === chat.active_image_profile_id)
    ?? profiles.find((profile) => profile.role === "image" && profile.is_default);
  const profileValues = {
    ...(imageProfile?.load_settings_json ?? {}),
    ...(imageProfile?.request_settings_json ?? {}),
  };
  const workflowSchema = workflowSchemaForTurn(
    workflows,
    project,
    mode,
    attachments.length > 0 || usePriorVisual,
  );

  const submit = (stopCurrent = false) => {
    if (!text.trim()) return;
    const selectedMode = currentMode();
    const role = roleForMode(selectedMode);
    const engine = engines.find((item) => item.roles.includes(role));
    const fields = resolveWorkflowSettings(
      resolveCapabilitySettings(engine, role),
      workflowSchema,
    );
    const dispatch = stopCurrent ? onStopAndSend : onSend;
    dispatch(
      text.trim(),
      selectedMode,
      attachments.map((item) => item.id),
      selectedMode === "auto"
        ? {}
        : normalizeSettingsForFields(
            templateSettings ? { ...settings, ...templateSettings.settings } : settings,
            fields,
          ),
    );
    setText("");
    setAttachments([]);
    setTemplateSettings(null);
  };

  return (
    <>
      <div
        className={`composer-wrap${dropActive ? " drop-active" : ""}`}
        onDragOver={(event) => {
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
        {attachments.length > 0 && (
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
                            onMode("image");
                            textInput.current?.focus();
                          }}
                        >
                          Edit
                        </button>
                        <button
                          className="attachment-edit"
                          aria-label="Animate attached image"
                          onClick={() => {
                            onMode("video");
                            setText((current) => current.trim() ? current : "Animate this image");
                            textInput.current?.focus();
                          }}
                        >
                          Animate
                        </button>
                      </>
                    )}
                    <button
                      aria-label={`Remove ${label}: ${name}`}
                      onClick={() => setAttachments((items) => (
                        items.filter((item) => item.id !== attachment.id)
                      ))}
                    >
                      <X size={12} />
                    </button>
                  </span>
                </article>
              );
            })}
          </div>
        )}
        {templateSettings && (
          <div className="template-settings-chip">
            <span>{templateSettings.name} settings apply to this send</span>
            <button aria-label="Remove template settings" onClick={() => setTemplateSettings(null)}><X size={12} /></button>
          </div>
        )}
        <div className="composer">
          <textarea
            ref={textInput}
            aria-label="Message"
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder="Ask anything, or describe an image or video to create…"
            rows={1}
          />
          <div className="composer-tools">
            <div className="left-tools">
              <button className="icon-button" onClick={() => fileInput.current?.click()} disabled={uploading} aria-label="Attach file"><Paperclip size={18} /></button>
              <input ref={fileInput} hidden multiple type="file" accept="image/*,video/*" onChange={(event) => { setUploadError(""); void uploadFiles(Array.from(event.target.files ?? [])); event.target.value = ""; }} />
              <button
                className="icon-button"
                onClick={() => setPromptHelperDraft(text.trim())}
                disabled={!text.trim()}
                aria-label="Open prompt workshop"
                title="Improve this prompt"
              >
                <Sparkles size={18} />
              </button>
              <label className={`mode-select mode-${mode}`}>
                {mode === "auto" && <Sparkles size={15} />}
                {mode === "text" && <MessageSquare size={15} />}
                {mode === "image" && <ImageIcon size={15} />}
                {mode === "video" && <Film size={15} />}
                <select aria-label="Generation mode" value={mode} onChange={(event) => changeMode(event.target.value as RoutingMode)}>
                  <option value="auto">Auto</option><option value="text">Text</option><option value="image">Image</option><option value="video">Video</option>
                </select>
                <ChevronDown size={13} />
              </label>
              {imageEdit && <button className="icon-button" onClick={() => setStudioOpen(true)} aria-label="Open editing studio" title="One-click edits"><Wand2 size={18} /></button>}
              <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Turn settings"><SlidersHorizontal size={18} /></button>
            </div>
            <span className="composer-submit-actions">
              {stoppable && (
                <button
                  className="send-button stop"
                  onClick={onStop}
                  aria-label="Stop current response"
                  title="Stop current response"
                >
                  <CircleStop size={18} />
                </button>
              )}
              {stoppable && text.trim() && (
                <button
                  className="secondary stop-and-send"
                  onClick={() => submit(true)}
                  aria-label="Stop current response and send"
                >
                  Stop and send
                </button>
              )}
              <button
                className="send-button"
                disabled={!text.trim()}
                onClick={() => submit()}
                aria-label="Send"
              >
                <Send size={18} />
              </button>
            </span>
          </div>
        </div>
      </div>
      {studioOpen && <EditingStudio currentInstruction={text} onClose={() => setStudioOpen(false)} onPick={(instruction, template) => { setText(instruction); setTemplateSettings(Object.keys(template.settings_json).length ? { name: template.name, settings: template.settings_json } : null); setStudioOpen(false); window.setTimeout(() => textInput.current?.focus(), 0); }} imageCount={attachments.filter((item) => item.kind === "image").length} onApplyToEach={(instruction, template) => {
        const role = roleForMode("image");
        const engine = engines.find((item) => item.roles.includes(role));
        const fields = resolveWorkflowSettings(resolveCapabilitySettings(engine, role), workflowSchema);
        const merged = normalizeSettingsForFields({ ...settings, ...template.settings_json }, fields);
        // One ordinary edit turn per image: each queues, verifies, and retries
        // alone; the pending-work bound errs clearly rather than truncating.
        for (const item of attachments.filter((entry) => entry.kind === "image")) onSend(instruction, "image", [item.id], merged);
        setAttachments([]); setText(""); setTemplateSettings(null); setStudioOpen(false);
      }} />}
      {promptHelperDraft !== null && (
        <PromptHelperDialog
          sourceChat={chat}
          initialDraft={promptHelperDraft}
          engines={engines}
          workflows={workflows}
          // Only explicit attachments ground the workshop: the helper chat has
          // no lineage, so a prior-image reference has nothing to resolve to.
          editSourceArtifactIds={imageEdit
            ? attachments.filter((item) => item.kind === "image").map((item) => item.id)
            : undefined}
          onAccept={(nextDraft) => {
            setText(nextDraft);
            setPromptHelperDraft(null);
            window.setTimeout(() => textInput.current?.focus(), 0);
          }}
          onClose={() => setPromptHelperDraft(null)}
        />
      )}
      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        mode={mode}
        engines={engines}
        values={settings}
        onValues={onSettings}
        presets={presets}
        presetId={presetId}
        onPreset={onPreset}
        workflowSchema={workflowSchema}
        inheritedValues={project?.generation_settings_json?.[roleForMode(mode)]}
        inheritedPresetId={project?.generation_preset_ids_json?.[roleForMode(mode)]}
        profileValues={profileValues}
        imageEdit={imageEdit}
        imageEditPrompt={text}
      />
    </>
  );
}

function activeBranchMessages(chat: ChatDetail): Message[] {
  const visibleMessages = chat.messages.filter(
    (message) => message.transcript_visible !== false,
  );
  if (!chat.active_head_message_id) return visibleMessages;
  const byId = new Map(visibleMessages.map((message) => [message.id, message]));
  const lineage: Message[] = [];
  const visited = new Set<string>();
  let current = byId.get(chat.active_head_message_id);
  while (current && !visited.has(current.id)) {
    visited.add(current.id);
    lineage.unshift(current);
    current = current.parent_id ? byId.get(current.parent_id) : undefined;
  }
  return lineage.length > 0 ? lineage : visibleMessages;
}

function workflowSchemaForTurn(
  workflows: Workflow[],
  project: Project | undefined,
  mode: RoutingMode,
  hasAttachments: boolean,
): Record<string, unknown> | undefined {
  if (mode !== "image" && mode !== "video") return undefined;
  const operation = mode === "image"
    ? hasAttachments ? "image_to_image" : "text_to_image"
    : hasAttachments ? "image_to_video" : "text_to_video";
  const pinnedRevisionId = mode === "image"
    ? project?.image_workflow_revision_id
    : project?.video_workflow_revision_id;
  if (pinnedRevisionId) {
    for (const workflow of workflows) {
      const revision = workflow.revisions.find((item) => item.id === pinnedRevisionId);
      if (revision && workflow.operation === operation) return revision.input_schema_json;
    }
  }
  const workflow = workflows.find((item) => item.operation === operation);
  return workflow?.revisions.find((item) => item.id === workflow.current_revision_id)
    ?.input_schema_json;
}

function ChatView({
  chat,
  engines,
  profiles,
  workflows,
  project,
  liveText,
  pendingTurns,
  workPlans,
  settings,
  presets,
  presetId,
  onSettings,
  onPreset,
  onMode,
  onSend,
  onProfile,
  onRegenerate,
  onSelectRevision,
  onEdit,
  onStop,
  onStopAndSend,
  onCancelPlan,
  onCancelStep,
  onRetryStep,
  onDeleteExchange,
  onForkThread,
  libraryEdit,
}: {
  chat?: ChatDetail;
  engines: EngineCapabilities[];
  profiles: ModelProfile[];
  workflows: Workflow[];
  project?: Project;
  liveText: Record<string, string>;
  pendingTurns: PendingTurn[];
  workPlans: WorkPlan[];
  settings: Record<string, unknown>;
  presets: GenerationPreset[];
  presetId: string | null;
  onSettings: (settings: Record<string, unknown>) => void;
  onPreset: (presetId: string | null) => void;
  onMode: (mode: RoutingMode) => void;
  onSend: (text: string, mode: RoutingMode, artifacts: string[], settings: Record<string, unknown>) => void;
  onProfile: (field: "active_chat_profile_id" | "active_vision_profile_id" | "active_image_profile_id" | "active_video_profile_id", id: string | null) => void;
  onRegenerate: (messageId: string, settings: Record<string, unknown>) => void;
  onSelectRevision: (messageId: string, revisionId: string) => void;
  onEdit: (
    messageId: string,
    text: string,
    mode: RoutingMode,
    settings: Record<string, unknown>,
  ) => void;
  onStop: () => void;
  onStopAndSend: (
    text: string,
    mode: RoutingMode,
    artifacts: string[],
    settings: Record<string, unknown>,
  ) => void;
  onCancelPlan: (planId: string) => void;
  onCancelStep: (stepId: string) => void;
  onRetryStep: (stepId: string) => void;
  onDeleteExchange: (messageId: string) => void;
  onForkThread: (messageId: string) => void;
  libraryEdit?: VisualTarget | null;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const followMessages = useRef(true);
  const previousChatId = useRef<string | undefined>(undefined);
  const [visualTarget, setVisualTarget] = useState<VisualTarget | null>(null);
  const favoriteClient = useQueryClient();
  const toggleFavorite = useMutation({
    mutationFn: ({ artifactId, next }: { artifactId: string; next: boolean }) =>
      api.favoriteArtifact(artifactId, next),
    onSuccess: () => {
      void favoriteClient.invalidateQueries({ queryKey: ["chat"] });
      void favoriteClient.invalidateQueries({ queryKey: ["artifacts"] });
    },
  });
  const consumedLibraryEdit = useRef<number | null>(null);
  useEffect(() => {
    if (!libraryEdit || consumedLibraryEdit.current === libraryEdit.requestId) return;
    consumedLibraryEdit.current = libraryEdit.requestId;
    setVisualTarget(libraryEdit);
  }, [libraryEdit]);
  const [quoteTarget, setQuoteTarget] = useState<{ text: string; requestId: number } | null>(null);
  useEffect(() => {
    if (previousChatId.current !== chat?.id) {
      previousChatId.current = chat?.id;
      followMessages.current = true;
    }
    if (followMessages.current && typeof endRef.current?.scrollIntoView === "function") {
      endRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chat?.id, chat?.messages, liveText, pendingTurns]);
  const trackMessageScroll = () => {
    const viewport = messagesRef.current;
    if (!viewport) return;
    followMessages.current = (
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight
    ) <= 96;
  };
  if (!chat) return <EmptyState icon={<MessageSquare />} title="Start a local conversation" body="Create a chat and choose a model. Conversations stay on this machine." />;
  const messages = activeBranchMessages(chat);
  const priorVisibleMedia = priorVisibleMediaByMessage(messages);
  const stoppable = messages.some(
    (message) => message.status === "pending"
      || (message.response_revisions ?? []).some(
        (revision) => revision.status === "pending",
      ),
  );
  const busy = stoppable || pendingTurns.length > 0;
  const planByAssistantMessage = new Map(
    workPlans.flatMap((plan) => {
      const assistantMessageIds = Array.isArray(plan.summary_json.assistant_message_ids)
        ? plan.summary_json.assistant_message_ids.filter(
          (messageId): messageId is string => typeof messageId === "string",
        )
        : [];
      const legacyAssistantMessageId = plan.summary_json.assistant_message_id;
      if (typeof legacyAssistantMessageId === "string") {
        assistantMessageIds.push(legacyAssistantMessageId);
      }
      return [...new Set(assistantMessageIds)].map(
        (messageId) => [messageId, plan] as const,
      );
    }),
  );
  return (
    <div className="chat-view">
      <div className="chat-header">
        <div><small>{chat.project_id ? "Project chat" : "Unfiled chat"}</small><h1>{chat.title}</h1></div>
        <div className="chat-profile-selectors">
          {(["chat", "vision", "image", "video"] as const).map((role) => {
            const field = `active_${role}_profile_id` as "active_chat_profile_id" | "active_vision_profile_id" | "active_image_profile_id" | "active_video_profile_id";
            const defaultProfile = role === "vision"
              ? undefined
              : profiles.find((profile) => profile.role === role && profile.is_default);
            const selected = profiles.find((profile) => profile.id === chat[field]);
            const value = role !== "vision" && selected?.is_default ? "" : chat[field] ?? "";
            const options = profiles.filter((profile) => (
              role === "vision"
                ? profile.role === "chat" && profile.input_modalities?.includes("image")
                : profile.role === role
            ) && (role === "vision" || !profile.is_default));
            return <label key={role}><span>{role}</span><select value={value} onChange={(event) => onProfile(field, event.target.value || null)}><option value={AUTO_PROFILE_ID}>Auto</option><option value="">{role === "vision" ? "Off" : `Default${defaultProfile ? ` · ${defaultProfile.name}` : ""}`}</option>{options.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}</select></label>;
          })}
        </div>
      </div>
      <div className="messages" ref={messagesRef} onScroll={trackMessageScroll}>
        {messages.length === 0 && pendingTurns.length === 0 ? (
          <EmptyState icon={<Sparkles />} title="What should we make?" body="Ask anything or create an image or video. Auto mode picks the model." />
        ) : messages.map((message, messageIndex) => {
          const messagePlan = planByAssistantMessage.get(message.id);
          const compareSourceUrl = message.role === "assistant"
            ? editSourceUrlForResult(messages, messageIndex)
            : null;
          const lineage = compareSourceUrl ? editLineageForResult(messages, messageIndex) : undefined;
          const isPrimaryOutput = messagePlan?.summary_json.assistant_message_id === message.id;
          return (
            <Fragment key={message.id}>
              {messagePlan && messagePlan.steps.length > 1 && isPrimaryOutput && (
                <MediaOutputPlan
                  plan={messagePlan}
                  onCancelStep={onCancelStep}
                  onRetryStep={onRetryStep}
                />
              )}
              <MessageBubble
                message={message}
                liveText={liveText[message.id]}
                compareSourceUrl={compareSourceUrl}
                lineage={lineage}
                onToggleFavorite={(part) => part.artifact_id && toggleFavorite.mutate({
                  artifactId: part.artifact_id,
                  next: !part.artifact?.favorite,
                })}
                hiddenInputArtifactIds={priorVisibleMedia.get(message.id)}
                onRegenerate={busy ? undefined : (messageId) => onRegenerate(
                  messageId,
                  chat.routing_mode === "auto" ? {} : settings,
                )}
                onSelectRevision={busy ? undefined : onSelectRevision}
                onEdit={busy ? undefined : (messageId, text) => onEdit(
                  messageId,
                  text,
                  chat.routing_mode,
                  chat.routing_mode === "auto" ? {} : settings,
                )}
                onCancelQueued={
                  messagePlan && messagePlan.steps.length <= 1 && messagePlan.status === "queued"
                    ? () => onCancelPlan(messagePlan.id)
                    : undefined
                }
                onEditImage={busy ? undefined : (part, origin) => setVisualTarget({
                  attachment: {
                    id: part.artifact_id!,
                    kind: "image",
                    artifact: part.artifact,
                    origin,
                  },
                  mode: "image",
                  requestId: Date.now(),
                })}
                onAnimateImage={busy ? undefined : (part, origin) => setVisualTarget({
                  attachment: {
                    id: part.artifact_id!,
                    kind: "image",
                    artifact: part.artifact,
                    origin,
                  },
                  mode: "video",
                  requestId: Date.now(),
                })}
                onReferenceMedia={busy ? undefined : (part, origin) => setVisualTarget({
                  attachment: {
                    id: part.artifact_id!,
                    kind: part.type === "video" ? "video" : "image",
                    artifact: part.artifact,
                    origin,
                  },
                  mode: null,
                  requestId: Date.now(),
                })}
                onQuote={(text) => setQuoteTarget({ text, requestId: Date.now() })}
                onDeleteExchange={busy ? undefined : onDeleteExchange}
                onForkThread={busy ? undefined : onForkThread}
              />
            </Fragment>
          );
        })}
        {pendingTurns.map((pendingTurn) => (
          <Fragment key={pendingTurn.id}>
            <article className="message user optimistic">
              <div className="avatar">You</div>
              <div className="message-content"><div className="message-text">{pendingTurn.text}</div></div>
            </article>
            <article
              className="message assistant optimistic"
              aria-live="polite"
            >
              <div className="avatar"><Bot size={19} /></div>
              <div className="message-content">
                <div className="submission-progress">
                  <LoaderCircle size={17} />
                  <span>{pendingTurn.mode === "auto" ? "Choosing mode and model…" : "Starting…"}</span>
                </div>
              </div>
            </article>
          </Fragment>
        ))}
        <div ref={endRef} />
      </div>
      <Composer chat={chat} engines={engines} profiles={profiles} stoppable={stoppable} settings={settings} onSettings={onSettings} presets={presets} presetId={presetId} onPreset={onPreset} onMode={onMode} onSend={onSend} onStop={onStop} onStopAndSend={onStopAndSend} workflows={workflows} project={project} visualTarget={visualTarget} quoteTarget={quoteTarget} />
    </div>
  );
}

function recipeOperationLabel(operation: string): string {
  return operation
    .split("_")
    .map((part) => part === "to" ? "→" : part)
    .join(" ")
    .replace(/^\w/, (character) => character.toUpperCase());
}
function RecipeCard({ recipe, pending, onInstall }: { recipe: ReferenceRecipe; pending: boolean; onInstall: () => void }) {
  const memory = recipe.hardware.minimum_vram_gb
    ? `${recipe.hardware.minimum_vram_gb} GB+ VRAM`
    : `${recipe.hardware.minimum_ram_gb} GB+ RAM`;
  return (
    <article className="recipe-card">
      <header>
        <div className="model-icon">{recipe.role === "video" ? <Film /> : recipe.role === "image" ? <ImageIcon /> : <Bot />}</div>
        <div><small>{recipe.role} · recipe v{recipe.version}</small><h3>{recipe.name}</h3></div>
      </header>
      <p>{recipe.summary}</p>
      <div className="recipe-badges"><span className={`badge ${recipe.certified ? "likely" : ""}`}>{recipe.certified ? "Certified" : "Reference candidate"}</span>{recipe.operations.map((operation) => <span className="badge" key={operation}>{recipeOperationLabel(operation)}</span>)}<span className="badge">{recipe.license_id}</span><span className="badge">{recipe.node_policy || recipe.engine}</span></div>
      <div className="recipe-meta"><span><HardDrive size={14} />{formatBytes(recipe.total_size_bytes)}</span><span><Gauge size={14} />{memory}</span></div>
      <small>{recipe.hardware.guidance}</small>
      <button className="primary" onClick={onInstall} disabled={pending}>{pending ? "Queued" : "Install recipe"}</button>
    </article>
  );
}

function InstalledModelRow({
  model,
  profile,
  creating,
  deleting,
  saving,
  defaulting,
  onCreate,
  onDelete,
  onSaveUseCase,
  onSetDefault,
}: {
  model: ModelInstall;
  profile?: ModelProfile;
  creating: boolean;
  deleting: boolean;
  saving: boolean;
  defaulting: boolean;
  onCreate: () => void;
  onDelete: () => void;
  onSaveUseCase: (value: string) => Promise<boolean>;
  onSetDefault: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(profile?.use_case ?? "");
  const startEditing = () => {
    setDraft(profile?.use_case ?? "");
    setEditing(true);
  };
  const save = async () => {
    if (await onSaveUseCase(draft.trim())) setEditing(false);
  };
  return (
    <div className={editing ? "editing" : ""}>
      <span className="badge">{model.role}</span>
      <span className="model-install-copy">
        <strong>{model.name}</strong>
        <small>{model.readiness === "ready" ? "Runtime verified" : model.readiness === "unsupported" ? "Unsupported" : "Not runtime verified"}</small>
        {model.role === "chat" && model.readiness === "ready" && profile?.input_modalities?.includes("text") && (
          <small>{profile.input_modalities.includes("image") ? "Vision capable" : "Text only"}</small>
        )}
        {profile?.use_case && <small>{profile.use_case}</small>}
      </span>
      <span className="model-install-size">{formatBytes(model.size_bytes)}</span>
      <span className="row-actions">
        {profile?.is_default
          ? <span className="badge tested">Default</span>
          : <button className="secondary compact-button" aria-label={`Set ${model.name} as default ${model.role} model`} disabled={creating || defaulting} onClick={onSetDefault}>{defaulting ? "Setting..." : "Set default"}</button>}
        {profile
          ? <button className="secondary compact-button" aria-label={`Edit use case for ${model.name}`} onClick={startEditing} disabled={editing || saving}>Edit use case</button>
          : <button className="secondary compact-button" aria-label={`Add ${model.name} to model selectors`} disabled={creating} onClick={onCreate}>Add to selectors</button>}
        <button className="secondary compact-button danger" aria-label={`Delete ${model.name}`} disabled={deleting} onClick={onDelete}>Delete</button>
      </span>
      {editing && profile && (
        <form className="model-use-case-editor" onSubmit={(event) => { event.preventDefault(); void save(); }}>
          <textarea aria-label={`Best uses for ${model.name}`} rows={2} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Programming, illustration, cinematic video…" />
          <button type="button" className="secondary compact-button" disabled={saving} onClick={() => setEditing(false)}>Cancel</button>
          <button type="submit" className="primary compact-button" disabled={saving || draft.trim() === profile.use_case.trim()}>{saving ? "Saving…" : "Save"}</button>
        </form>
      )}
    </div>
  );
}

type ModelAssetUpdateValues = Partial<Pick<
  ModelAssetInstall,
  "active" | "use_case" | "auto_apply" | "default_model_strength" | "default_clip_strength"
>>;

function InstalledAssetRow({
  asset,
  saving,
  deleting,
  onUpdate,
  onDelete,
}: {
  asset: ModelAssetInstall;
  saving: boolean;
  deleting: boolean;
  onUpdate: (values: ModelAssetUpdateValues) => Promise<boolean>;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [useCase, setUseCase] = useState(asset.use_case);
  const [autoApply, setAutoApply] = useState(asset.auto_apply);
  const [modelStrength, setModelStrength] = useState(String(asset.default_model_strength));
  const [clipStrength, setClipStrength] = useState(String(asset.default_clip_strength));
  const beginEditing = () => {
    setUseCase(asset.use_case);
    setAutoApply(asset.auto_apply);
    setModelStrength(String(asset.default_model_strength));
    setClipStrength(String(asset.default_clip_strength));
    setEditing(true);
  };
  const parsedModelStrength = Number(modelStrength);
  const parsedClipStrength = Number(clipStrength);
  const strengthsValid = Number.isFinite(parsedModelStrength)
    && Math.abs(parsedModelStrength) <= 4
    && Number.isFinite(parsedClipStrength)
    && Math.abs(parsedClipStrength) <= 4;
  const unchanged = useCase.trim() === asset.use_case
    && autoApply === asset.auto_apply
    && parsedModelStrength === asset.default_model_strength
    && parsedClipStrength === asset.default_clip_strength;
  const save = async () => {
    if (!strengthsValid || (autoApply && !useCase.trim())) return;
    const saved = await onUpdate({
      use_case: useCase.trim(),
      auto_apply: autoApply,
      default_model_strength: parsedModelStrength,
      default_clip_strength: parsedClipStrength,
    });
    if (saved) setEditing(false);
  };
  return (
    <div className={editing ? "editing" : ""}>
      <span className="badge">{asset.kind.replace("_", " ")}</span>
      <span className="model-install-copy">
        <strong>{asset.name}</strong>
        <small>{asset.active ? "Ready" : "Disabled"}{asset.family ? ` · ${asset.family}` : ""}</small>
        {asset.kind === "lora" && asset.auto_apply && asset.use_case && (
          <small>Auto · {asset.use_case}</small>
        )}
      </span>
      <span className="model-install-size">{formatBytes(asset.size_bytes)}</span>
      <span className="row-actions">
        <button
          className="secondary compact-button"
          disabled={!asset.verified_at || saving}
          onClick={() => void onUpdate({ active: !asset.active })}
        >
          {asset.active ? "Disable" : "Enable"}
        </button>
        {asset.kind === "lora" && (
          <button className="secondary compact-button" disabled={editing || saving} onClick={beginEditing}>
            Edit Auto rules
          </button>
        )}
        <button className="secondary compact-button danger" disabled={deleting} onClick={onDelete}>Delete</button>
      </span>
      {editing && asset.kind === "lora" && (
        <form className="model-use-case-editor lora-auto-editor" onSubmit={(event) => { event.preventDefault(); void save(); }}>
          <label>
            Use case
            <textarea aria-label={`Auto use case for ${asset.name}`} rows={2} value={useCase} onChange={(event) => setUseCase(event.target.value)} placeholder="Watercolor landscapes, product photography…" />
          </label>
          <label>
            Model strength
            <input aria-label={`Default model strength for ${asset.name}`} type="number" min="-4" max="4" step="0.05" value={modelStrength} onChange={(event) => setModelStrength(event.target.value)} />
          </label>
          <label>
            CLIP strength
            <input aria-label={`Default CLIP strength for ${asset.name}`} type="number" min="-4" max="4" step="0.05" value={clipStrength} onChange={(event) => setClipStrength(event.target.value)} />
          </label>
          <label className="lora-auto-toggle">
            <input aria-label={`Use ${asset.name} automatically`} type="checkbox" checked={autoApply} onChange={(event) => setAutoApply(event.target.checked)} />
            Use automatically
          </label>
          <span className="row-actions">
            <button type="button" className="secondary compact-button" disabled={saving} onClick={() => setEditing(false)}>Cancel</button>
            <button type="submit" className="primary compact-button" disabled={saving || unchanged || !strengthsValid || (autoApply && !useCase.trim())}>{saving ? "Saving…" : "Save"}</button>
          </span>
        </form>
      )}
    </div>
  );
}

interface PendingInstall {
  model: CatalogModel;
  preflight: CatalogPreflight;
  installRole: string;
  engine: string;
  auxiliaryKind: "lora" | null;
}

function ModelsView({ initialRole }: { initialRole: EngineRole }) {
  const client = useQueryClient();
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [role, setRole] = useState<string>(initialRole);
  const [sort, setSort] = useState("trending");
  const [compatibility, setCompatibility] = useState("");
  const [quantization, setQuantization] = useState("");
  const [maxSizeGb, setMaxSizeGb] = useState("");
  const [updatedWithinDays, setUpdatedWithinDays] = useState("");
  const [installedChatCapability, setInstalledChatCapability] = useState("");
  const [importOpen, setImportOpen] = useState(false);
  const [importName, setImportName] = useState("");
  const [importPath, setImportPath] = useState("");
  const [importRole, setImportRole] = useState("chat");
  const [importEngine, setImportEngine] = useState("llama.cpp");
  const catalogFilters = {
    compatibility,
    quantization,
    max_size_bytes: maxSizeGb ? String(Number(maxSizeGb) * 1024 ** 3) : "",
    updated_within_days: updatedWithinDays,
  };
  const catalog = useInfiniteQuery({
    queryKey: ["catalog", submitted, role, sort, catalogFilters],
    queryFn: ({ pageParam }) => api.catalog(submitted, role, sort, pageParam, catalogFilters),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });
  const rawCatalogItems = useMemo(() => catalog.data?.pages.flatMap((page) => page.items) ?? [], [catalog.data]);
  const catalogItems = rawCatalogItems;
  const catalogIsStale = catalog.data?.pages.some((page) => page.stale) ?? false;
  const recipes = useQuery({ queryKey: ["recipes"], queryFn: api.recipes });
  const installed = useQuery({ queryKey: ["models"], queryFn: api.models });
  const modelAssets = useQuery({ queryKey: ["model-assets"], queryFn: () => api.modelAssets() });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3_000 });
  const storage = useQuery({ queryKey: ["model-storage"], queryFn: api.modelStorage });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const runtimes = useQuery({ queryKey: ["runtimes"], queryFn: api.runtimes });
  const machine = useQuery({ queryKey: ["system"], queryFn: api.system });
  const runtimeFor = (model: CatalogModel) => runtimes.data?.find(
    (runtime) => runtime.engine === model.required_runtime,
  );
  // Preflight and transfer are separate steps so the user sees what a download
  // will cost before it starts; the numbers were previously computed and dropped.
  const [pendingInstall, setPendingInstall] = useState<PendingInstall | null>(null);
  const download = useMutation({
    mutationFn: async ({ model, selectedRole }: { model: CatalogModel; selectedRole: string }) => {
      const auxiliaryKind = selectedRole === "lora" ? "lora" : null;
      const installRole = auxiliaryKind ? "image" : selectedRole;
      const engine = model.required_runtime ?? (installRole === "chat" ? "llama.cpp" : "comfyui");
      const preflight = auxiliaryKind
        ? await api.catalogPreflight(
            model.remote_id,
            installRole,
            engine,
            "main",
            [],
            auxiliaryKind,
          )
        : await api.catalogPreflight(
            model.remote_id,
            installRole,
            engine,
            "main",
            [],
            null,
            // Preflight the exact workflow this card represents; a repository
            // can ship several and ranking must not answer for the user.
            model.workflow_template_id ?? null,
          );
      if (!preflight.can_install) {
        const blockers = preflight.checks
          .filter((check) => check.status === "block")
          .map((check) => check.detail);
        throw new Error(blockers.join(" ") || "This model cannot be installed safely.");
      }
      if (!preflight.install_plan || preflight.install_plan.compatibility !== "supported") {
        throw new Error(
          preflight.install_plan?.failure_reason
          || "LM Atelier cannot safely activate this model with the current runtime.",
        );
      }
      return { model, preflight, installRole, engine, auxiliaryKind } satisfies PendingInstall;
    },
    onSuccess: (ready) => setPendingInstall(ready),
  });
  const confirmInstall = useMutation({
    mutationFn: ({ preflight, installRole, engine, auxiliaryKind }: PendingInstall) => {
      const downloadArguments = [
        preflight.remote_id,
        preflight.source_remote_id,
        installRole,
        engine,
        preflight.revision,
        preflight.selected_files,
        preflight.expected_sha256,
        preflight.file_sources ?? {},
        preflight.comfy_paths,
        preflight.workflow_template_id,
        preflight.workflow_template_sha256,
        preflight.install_plan?.id ?? null,
      ] as const;
      const contentRating = preflight.content_rating ?? "unknown";
      return auxiliaryKind
        ? api.download(...downloadArguments, auxiliaryKind, contentRating)
        : api.download(...downloadArguments, null, contentRating);
    },
    onSuccess: () => {
      setPendingInstall(null);
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const installRecipe = useMutation({
    mutationFn: (recipeId: string) => api.installRecipe(recipeId),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const createProfile = useMutation({
    mutationFn: (model: ModelInstall) => api.createProfile(model),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["profiles"] }),
  });
  const updateUseCase = useMutation({
    mutationFn: ({ profileId, useCase }: { profileId: string; useCase: string }) =>
      api.updateProfile(profileId, { use_case: useCase }),
    onSuccess: (updated) => {
      client.setQueryData<ModelProfile[]>(["profiles"], (current) =>
        current?.map((profile) => profile.id === updated.id ? updated : profile) ?? [updated],
      );
      void client.invalidateQueries({ queryKey: ["profiles"] });
    },
  });
  const setDefaultModel = useMutation({
    mutationFn: ({ model, profile }: { model: ModelInstall; profile?: ModelProfile }) =>
      profile
        ? api.updateProfile(profile.id, { is_default: true })
        : api.createProfile(model, true),
    onSuccess: (updated) => {
      client.setQueryData<ModelProfile[]>(["profiles"], (current) => {
        const siblings = (current ?? []).map((profile) => (
          profile.role === updated.role
            ? { ...profile, is_default: profile.id === updated.id }
            : profile
        ));
        return siblings.some((profile) => profile.id === updated.id)
          ? siblings
          : [...siblings, updated];
      });
      void client.invalidateQueries({ queryKey: ["profiles"] });
    },
  });
  const deleteModel = useMutation({
    mutationFn: (modelId: string) => api.deleteModel(modelId, true),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["profiles"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const updateModelAsset = useMutation({
    mutationFn: ({ id, values }: { id: string; values: ModelAssetUpdateValues }) =>
      api.updateModelAsset(id, values),
    onSuccess: (updated) => {
      client.setQueryData<ModelAssetInstall[]>(["model-assets"], (current) =>
        current?.map((asset) => asset.id === updated.id ? updated : asset) ?? [updated],
      );
      void client.invalidateQueries({ queryKey: ["model-assets"] });
    },
  });
  const deleteModelAsset = useMutation({
    mutationFn: api.deleteModelAsset,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["model-assets"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const cleanupDownloads = useMutation({
    mutationFn: api.cleanupDownloads,
    onSuccess: () => void client.invalidateQueries({ queryKey: ["model-storage"] }),
  });
  const importModel = useMutation({
    mutationFn: () => api.importModel({ name: importName, local_path: importPath, role: importRole, engine: importEngine }),
    onSuccess: () => {
      setImportOpen(false);
      setImportName("");
      setImportPath("");
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const installedRemoteIds = new Set(
    (
      role === "lora"
        ? modelAssets.data
          ?.filter((asset) => asset.kind === "lora" && asset.active)
          .map((asset) => asset.manifest_json.remote_id)
        : installed.data
          ?.filter((model) => model.role === role && model.active)
          ?.flatMap((model) => [
            model.manifest_json.remote_id,
            model.manifest_json.source_remote_id,
          ])
    )?.filter((remoteId): remoteId is string => typeof remoteId === "string") ?? [],
  );
  const activeDownloadIds = new Set(
    jobs.data
      ?.filter((job) =>
        job.kind === "download"
        && (
          role === "lora"
            ? job.payload_json.auxiliary_kind === "lora"
            : job.payload_json.role === role && !job.payload_json.auxiliary_kind
        )
        && ["queued", "running", "paused"].includes(job.status)
      )
      .map((job) => job.payload_json.remote_id)
      .filter((remoteId): remoteId is string => typeof remoteId === "string") ?? [],
  );
  const installedModels = (installed.data ?? []).filter((model) => {
    if (!installedChatCapability) return true;
    if (model.role !== "chat" || model.readiness !== "ready") return false;
    const profile = profiles.data?.find((candidate) => candidate.model_install_id === model.id);
    const modalities = profile?.input_modalities ?? [];
    return installedChatCapability === "vision"
      ? modalities.includes("image")
      : modalities.includes("text") && !modalities.includes("image");
  });
  const installedTemplateIds = new Set(installed.data?.filter((model) => model.role === role && model.active).map((model) => model.manifest_json.workflow_template_id).filter((value): value is string => typeof value === "string") ?? []);
  // A workflow card is installed only when ITS template is: variants share one
  // repository, and installing one must not disable the others.
  const statusFor = (model: CatalogModel): "idle" | "preparing" | "downloading" | "installed" => (
    (model.workflow_template_id ? installedTemplateIds.has(model.workflow_template_id) : installedRemoteIds.has(model.remote_id))
      ? "installed"
      : activeDownloadIds.has(model.remote_id)
        ? "downloading"
        : download.isPending && download.variables?.model.remote_id === model.remote_id && (download.variables?.model.workflow_template_id ?? null) === (model.workflow_template_id ?? null)
          ? "preparing"
          : "idle"
  );
  return (
    <div className="page-view">
      <header className="page-header"><div><h1>Model library</h1></div><div className="storage-actions"><div className="storage-pill"><HardDrive size={17} />{storage.data?.installed_count ?? installed.data?.length ?? 0} installed · {formatBytes(storage.data?.installed_bytes)}</div><button className="secondary compact-button" onClick={() => setImportOpen(true)}><Folder size={16} />Import local</button>{Boolean(storage.data?.partial_download_count) && <button className="secondary compact-button" disabled={cleanupDownloads.isPending} onClick={() => cleanupDownloads.mutate()}>Clean {storage.data?.partial_download_count} partial</button>}</div></header>
      <ModelUpdatesPanel onInstall={(model, selectedRole) => download.mutate({ model, selectedRole })} />
      <section className="recipe-section">
        <div className="section-heading"><div><h2>Reference recipes</h2></div></div>
        {recipes.isLoading && <div className="loading-line" />}
        {recipes.error && <ErrorCallout message={recipes.error.message} />}
        {installRecipe.error && <ErrorCallout message={installRecipe.error.message} />}
        <div className="recipe-grid">{recipes.data?.map((recipe) => <RecipeCard key={recipe.id} recipe={recipe} pending={installRecipe.isPending && installRecipe.variables === recipe.id} onInstall={() => installRecipe.mutate(recipe.id)} />)}</div>
      </section>
      <div className="toolbar">
        <form className="search-box" onSubmit={(event) => { event.preventDefault(); setSubmitted(query); }}><Search size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models" /></form>
        <select aria-label="Model role" value={role} onChange={(event) => setRole(event.target.value)}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option><option value="lora">LoRA</option></select>
        <select aria-label="Model order" value={sort} onChange={(event) => setSort(event.target.value)}><option value="trending">Trending</option><option value="downloads">Downloads</option><option value="likes">Likes</option><option value="newest">Newest</option><option value="updated">Recently updated</option><option value="compatible">Compatible first</option></select>
      </div>
      <div className="catalog-filters"><select aria-label="Compatibility filter" value={compatibility} onChange={(event) => setCompatibility(event.target.value)}><option value="">All compatibility</option><option value="likely">Automatic test available</option><option value="advanced_import">Advanced import</option><option value="unsupported">Unsupported</option></select><select aria-label="Last updated filter" value={updatedWithinDays} onChange={(event) => setUpdatedWithinDays(event.target.value)}><option value="">Updated any time</option><option value="7">Updated this week</option><option value="30">Updated this month</option><option value="90">Updated in 3 months</option><option value="365">Updated this year</option></select><input aria-label="Quantization filter" placeholder="Quantization (Q4_K_M, FP8…)" value={quantization} onChange={(event) => setQuantization(event.target.value)} /><input aria-label="Maximum download size" type="number" min="0" placeholder="Max download (GB)" value={maxSizeGb} onChange={(event) => setMaxSizeGb(event.target.value)} /></div>
      {(installed.data?.length ?? 0) > 0 && <section>
        <div className="section-heading">
          <h2>Installed models</h2>
          <select
            aria-label="Installed chat capability"
            value={installedChatCapability}
            onChange={(event) => setInstalledChatCapability(event.target.value)}
          >
            <option value="">All capabilities</option>
            <option value="text">Text only</option>
            <option value="vision">Vision capable</option>
          </select>
        </div>
        <div className="profile-table model-installs">{installedModels.map((model) => {
        const profile = profiles.data?.find((candidate) => candidate.model_install_id === model.id);
        return <InstalledModelRow
          key={model.id}
          model={model}
          profile={profile}
          creating={createProfile.isPending && createProfile.variables?.id === model.id}
          deleting={deleteModel.isPending && deleteModel.variables === model.id}
          saving={updateUseCase.isPending && updateUseCase.variables?.profileId === profile?.id}
          defaulting={setDefaultModel.isPending && setDefaultModel.variables?.model.id === model.id}
          onCreate={() => createProfile.mutate(model)}
          onDelete={() => { if (window.confirm(`Delete ${model.name} and its model settings from local storage?`)) deleteModel.mutate(model.id); }}
          onSaveUseCase={async (value) => {
            if (!profile) return false;
            try {
              await updateUseCase.mutateAsync({ profileId: profile.id, useCase: value });
              return true;
            } catch {
              return false;
            }
          }}
          onSetDefault={() => setDefaultModel.mutate({ model, profile })}
        />;
        })}</div>
      </section>}
      {(modelAssets.data?.length ?? 0) > 0 && <section>
        <div className="section-heading"><h2>Installed workflow assets</h2></div>
        <div className="profile-table model-installs">
          {modelAssets.data?.map((asset) => (
            <InstalledAssetRow
              key={asset.id}
              asset={asset}
              saving={updateModelAsset.isPending && updateModelAsset.variables?.id === asset.id}
              deleting={deleteModelAsset.isPending && deleteModelAsset.variables === asset.id}
              onUpdate={async (values) => {
                try {
                  await updateModelAsset.mutateAsync({ id: asset.id, values });
                  return true;
                } catch {
                  return false;
                }
              }}
              onDelete={() => window.confirm(`Delete ${asset.name}?`) && deleteModelAsset.mutate(asset.id)}
            />
          ))}
        </div>
      </section>}
      {pendingInstall && (
        <InstallConfirmDialog
          name={pendingInstall.model.name || pendingInstall.model.remote_id}
          preflight={pendingInstall.preflight}
          system={machine.data}
          pending={confirmInstall.isPending}
          onConfirm={() => confirmInstall.mutate(pendingInstall)}
          onCancel={() => setPendingInstall(null)}
        />
      )}
      {(download.error || confirmInstall.error || deleteModel.error || cleanupDownloads.error || updateUseCase.error || setDefaultModel.error || updateModelAsset.error || deleteModelAsset.error) && <ErrorCallout message={download.error?.message || confirmInstall.error?.message || deleteModel.error?.message || cleanupDownloads.error?.message || updateUseCase.error?.message || setDefaultModel.error?.message || updateModelAsset.error?.message || deleteModelAsset.error?.message} />}
      {catalog.isLoading && <div className="loading-line" />}
      <ErrorCallout message={catalog.error?.message} action={<button className="secondary compact-button" disabled={catalog.isFetching} onClick={() => void catalog.refetch()}>Retry</button>} />
      {catalogIsStale && !catalog.error && <div className="callout warning action-callout" role="status"><span>Showing saved results while Hugging Face is unavailable.</span><button className="secondary compact-button" disabled={catalog.isFetching} onClick={() => void catalog.refetch()}>Refresh</button></div>}
      <div className="model-grid">{catalogItems.map((model) => <ModelCard key={model.remote_id} model={model} role={role} runtime={runtimeFor(model)} status={statusFor(model)} onDownload={() => download.mutate({ model, selectedRole: role })} />)}</div>
      {catalog.hasNextPage && <div className="load-more"><button className="secondary" disabled={catalog.isFetchingNextPage} onClick={() => void catalog.fetchNextPage()}>{catalog.isFetchingNextPage ? "Loading…" : "Load more models"}</button></div>}
      {importOpen && (
        <AccessibleDialog
          title="Import a local model"
          eyebrow="Advanced import"
          closeLabel="Close local import"
          onClose={() => setImportOpen(false)}
        >
          <p>Register a local file or folder. Pickle-compatible formats are blocked as unsafe, and imports require review before use.</p>
          <label>Display name<input value={importName} onChange={(event) => setImportName(event.target.value)} /></label>
          <label>Absolute local path<input value={importPath} onChange={(event) => setImportPath(event.target.value)} placeholder="/path/to/model.gguf" /></label>
          <label>Role<select value={importRole} onChange={(event) => { const next = event.target.value; setImportRole(next); setImportEngine(next === "chat" ? "llama.cpp" : "comfyui"); }}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option></select></label>
          <label>Runtime<select value={importEngine} onChange={(event) => setImportEngine(event.target.value)}><option value="llama.cpp">llama.cpp</option><option value="vllm">vLLM (ModelOpt/NVFP4)</option><option value="comfyui">ComfyUI</option></select></label>
          {importModel.error && <ErrorCallout message={importModel.error.message} />}
          <footer><button className="secondary" onClick={() => setImportOpen(false)}>Cancel</button><button className="primary" disabled={!importName.trim() || !importPath.trim() || importModel.isPending} onClick={() => importModel.mutate()}>{importModel.isPending ? "Importing…" : "Import model"}</button></footer>
        </AccessibleDialog>
      )}
    </div>
  );
}

function WorkflowControls({ schema }: { schema: Record<string, unknown> }) {
  const properties = schema.properties && typeof schema.properties === "object"
    ? schema.properties as Record<string, Record<string, unknown>>
    : {};
  const schemaKey = JSON.stringify(schema);
  const defaults = Object.fromEntries(
    Object.entries(properties).map(([key, field]) => [key, field.default ?? ""]),
  );
  const [stored, setStored] = useState<{ schemaKey: string; values: Record<string, unknown> }>(
    { schemaKey, values: defaults },
  );
  const values = stored.schemaKey === schemaKey ? stored.values : defaults;
  if (!Object.keys(properties).length) return <p className="muted">This revision does not declare user-facing inputs.</p>;
  return (
    <div className="workflow-controls">
      {Object.entries(properties).map(([key, field]) => {
        const label = String(field.title ?? key.replaceAll("_", " "));
        const description = typeof field.description === "string" ? field.description : "";
        const choices = Array.isArray(field.enum) ? field.enum : [];
        const type = String(field.type ?? "string");
        const update = (value: unknown) => setStored((current) => ({
          schemaKey,
          values: {
            ...(current.schemaKey === schemaKey ? current.values : defaults),
            [key]: value,
          },
        }));
        return (
          <label key={key}>
            <span><strong>{label}</strong>{description && <small>{description}</small>}</span>
            {choices.length ? (
              <select value={String(values[key] ?? "")} onChange={(event) => update(event.target.value)}>
                {choices.map((choice) => <option key={String(choice)} value={String(choice)}>{String(choice)}</option>)}
              </select>
            ) : type === "boolean" ? (
              <input type="checkbox" checked={Boolean(values[key])} onChange={(event) => update(event.target.checked)} />
            ) : (
              <input type={type === "integer" || type === "number" ? "number" : "text"} min={typeof field.minimum === "number" ? field.minimum : undefined} max={typeof field.maximum === "number" ? field.maximum : undefined} step={type === "integer" ? 1 : typeof field.multipleOf === "number" ? field.multipleOf : undefined} value={String(values[key] ?? "")} onChange={(event) => update(type === "integer" || type === "number" ? Number(event.target.value) : event.target.value)} />
            )}
          </label>
        );
      })}
    </div>
  );
}

function WorkflowsView() {
  const client = useQueryClient();
  const workflows = useQuery({ queryKey: ["workflows"], queryFn: api.workflows });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = workflows.data?.find((workflow) => workflow.id === selectedId) ?? null;
  const [selectedRevisionId, setSelectedRevisionId] = useState<string | null>(null);
  const [newOpen, setNewOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState("Custom image workflow");
  const [description, setDescription] = useState("");
  const [operation, setOperation] = useState("text_to_image");
  const [graph, setGraph] = useState("{}");
  const [uiGraph, setUiGraph] = useState("{}");
  const [inputSchema, setInputSchema] = useState("{}");
  const [dependencies, setDependencies] = useState("{}");
  const [trusted, setTrusted] = useState(false);
  const importInput = useRef<HTMLInputElement>(null);
  const refresh = () => void client.invalidateQueries({ queryKey: ["workflows"] });
  const save = useMutation({
    mutationFn: async () => {
      const revision = { engine_version: null, api_graph: JSON.parse(graph), ui_graph: JSON.parse(uiGraph), input_schema: JSON.parse(inputSchema), dependencies: JSON.parse(dependencies), trusted };
      if (editing && selected) {
        await api.updateWorkflow(selected.id, { name, description });
        return api.createWorkflowRevision(selected.id, revision);
      }
      return api.createWorkflow({ name, description, operation, engine: "comfyui", ...revision });
    },
    onSuccess: () => { setNewOpen(false); setEditing(false); refresh(); },
  });
  const validate = useMutation({ mutationFn: (id: string) => api.validateWorkflow(id) });
  const clone = useMutation({ mutationFn: (id: string) => api.cloneWorkflow(id), onSuccess: refresh });
  const restore = useMutation({ mutationFn: ({ id, revisionId }: { id: string; revisionId: string }) => api.restoreWorkflowRevision(id, revisionId), onSuccess: refresh });
  const exportBundle = useMutation({ mutationFn: (id: string) => api.exportWorkflow(id), onSuccess: (bundle) => downloadJson(bundle, `${bundle.name.replaceAll(/[^a-z0-9]+/gi, "-").toLowerCase()}.lm-atelier-workflow.json`) });
  const openInComfy = useMutation({ mutationFn: (id: string) => api.workflowOpenTarget(id), onSuccess: (target) => { downloadJson(target.ui_graph, target.filename); window.open(target.url, "_blank", "noopener,noreferrer"); } });
  const {
    importFile: importBundle,
    importError,
    packageReview,
    closePackageReview,
  } = useWorkflowPackageImport(refresh);
  const openCreate = () => { setEditing(false); setName("Custom image workflow"); setDescription(""); setOperation("text_to_image"); setGraph("{}"); setUiGraph("{}"); setInputSchema("{}"); setDependencies("{}"); setTrusted(false); setNewOpen(true); };
  const openEdit = () => { if (!selected) return; const revision = selected.revisions.find((item) => item.id === selected.current_revision_id) ?? selected.revisions.at(-1); if (!revision) return; setEditing(true); setName(selected.name); setDescription(selected.description); setOperation(selected.operation); setGraph(JSON.stringify(revision.api_graph_json, null, 2)); setUiGraph(JSON.stringify(revision.ui_graph_json, null, 2)); setInputSchema(JSON.stringify(revision.input_schema_json, null, 2)); setDependencies(JSON.stringify(revision.dependencies_json, null, 2)); setTrusted(revision.trusted); setNewOpen(true); };
  const selectedRevision = selected?.revisions.find((revision) => revision.id === selectedRevisionId) ?? selected?.revisions.find((revision) => revision.id === selected.current_revision_id) ?? selected?.revisions.at(-1);
  const currentRevision = selected?.revisions.find((revision) => revision.id === selected.current_revision_id);
  return (
    <div className="page-view">
      <header className="page-header"><div><h1>Workflows</h1></div><div className="storage-actions"><input ref={importInput} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0]); event.target.value = ""; }} /><button className="secondary" onClick={() => importInput.current?.click()}>Import bundle</button><button className="primary" onClick={openCreate}><Plus size={17} />New workflow</button></div></header>
      {(importError || clone.error || restore.error || exportBundle.error || openInComfy.error) && <ErrorCallout message={(importError || clone.error || restore.error || exportBundle.error || openInComfy.error)?.message} />}
      {packageReview && <WorkflowPackageReview analysis={packageReview.analysis} fileName={packageReview.fileName} uiGraph={packageReview.uiGraph} onImported={() => { closePackageReview(); refresh(); }} onClose={closePackageReview} />}
      {selected && <div className="storage-actions"><button className="secondary" onClick={() => openInComfy.mutate(selected.id)}>Download UI graph and open in ComfyUI</button></div>}
      <div className="workflow-layout">
        <div className="workflow-list">{workflows.data?.map((workflow) => <button key={workflow.id} className={selected?.id === workflow.id ? "selected" : ""} onClick={() => { setSelectedId(workflow.id); setSelectedRevisionId(workflow.current_revision_id); }}><WorkflowIcon size={18} /><span><strong>{workflow.name}</strong><small>{workflow.operation} · {workflow.revisions.length} revision{workflow.revisions.length === 1 ? "" : "s"}</small></span></button>)}</div>
        <div className="workflow-detail">{selected && selectedRevision ? <><div className="detail-title"><div><small>{selected.operation}</small><h2>{selected.name}</h2><p>{selected.description}</p></div><div className="row-actions"><button className="secondary compact-button" onClick={openEdit}>New revision</button><button className="secondary compact-button" onClick={() => clone.mutate(selected.id)}>Duplicate</button><button className="secondary compact-button" onClick={() => exportBundle.mutate(selected.id)}>Export</button><button className="secondary compact-button" onClick={() => validate.mutate(selected.id)}>Validate</button></div></div><div className="workflow-revision-bar"><label>Revision<select value={selectedRevision.id} onChange={(event) => setSelectedRevisionId(event.target.value)}>{[...selected.revisions].sort((a, b) => b.version - a.version).map((revision) => <option key={revision.id} value={revision.id}>v{revision.version}{revision.id === selected.current_revision_id ? " · current" : ""}</option>)}</select></label>{selectedRevision.id !== selected.current_revision_id && <button className="secondary compact-button" onClick={() => restore.mutate({ id: selected.id, revisionId: selectedRevision.id })}>Restore as new revision</button>}<span className={`badge ${selectedRevision.trusted ? "likely" : "advanced_import"}`}>{selectedRevision.trusted ? "Trusted" : "Untrusted"}</span></div><section className="workflow-input-section"><h3>Declared controls</h3><WorkflowControls schema={selectedRevision.input_schema_json} /></section><details open><summary>Executable graph</summary><pre>{JSON.stringify(selectedRevision.api_graph_json, null, 2)}</pre></details><details><summary>Dependencies</summary><pre>{JSON.stringify(selectedRevision.dependencies_json, null, 2)}</pre></details>{currentRevision && currentRevision.id !== selectedRevision.id && <details><summary>Compare with current revision</summary><div className="workflow-compare"><pre>{JSON.stringify(selectedRevision.api_graph_json, null, 2)}</pre><pre>{JSON.stringify(currentRevision.api_graph_json, null, 2)}</pre></div></details>}{validate.data && <div className={`callout ${validate.data.valid ? "success" : "error"}`} role={validate.data.valid ? "status" : "alert"}>{validate.data.valid ? "Workflow and declared dependencies are valid for the active media engine." : validate.data.errors.join("\n")}{validate.data.warnings.map((warning) => `\nWarning: ${warning}`)}</div>}</> : <EmptyState icon={<WorkflowIcon />} title="Select a workflow" body="Review its revision, inputs, dependencies, and validation." />}</div>
      </div>
      <RegistryInstallsPanel />
      <CustomNodesPanel />
      {newOpen && (
        <AccessibleDialog
          title={editing ? "Create workflow revision" : "Create ComfyUI workflow"}
          eyebrow={editing ? "Immutable revision" : "Portable workflow"}
          closeLabel="Close workflow editor"
          onClose={() => setNewOpen(false)}
          className="workflow-editor"
        >
          <label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <label>Description<textarea rows={2} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
          <label>Operation<select value={operation} disabled={editing} onChange={(event) => setOperation(event.target.value)}><option value="text_to_image">Text to image</option><option value="image_to_image">Image to image</option><option value="text_to_video">Text to video</option><option value="image_to_video">Image to video</option></select></label>
          <label>API-format workflow JSON<textarea rows={10} value={graph} onChange={(event) => setGraph(event.target.value)} /></label>
          <label>UI workflow JSON<textarea rows={5} value={uiGraph} onChange={(event) => setUiGraph(event.target.value)} /></label>
          <label>Declared input schema JSON<textarea rows={6} value={inputSchema} onChange={(event) => setInputSchema(event.target.value)} /></label>
          <label>Dependencies JSON<textarea rows={5} value={dependencies} onChange={(event) => setDependencies(event.target.value)} /></label>
          <label className="toggle-row"><span><strong>Trust this workflow</strong><small>Only enable after reviewing every node and dependency.</small></span><input type="checkbox" checked={trusted} onChange={(event) => setTrusted(event.target.checked)} /></label>
          {save.error && <ErrorCallout message={save.error.message} />}
          <footer><button className="secondary" onClick={() => setNewOpen(false)}>Cancel</button><button className="primary" disabled={!name.trim() || save.isPending} onClick={() => save.mutate()}>{save.isPending ? "Saving…" : editing ? "Create revision" : "Save workflow"}</button></footer>
        </AccessibleDialog>
      )}
    </div>
  );
}

function ProfileEditor({
  profile,
  engines,
  onClose,
}: {
  profile: ModelProfile;
  engines: EngineCapabilities[];
  onClose: () => void;
}) {
  const client = useQueryClient();
  const [name, setName] = useState(profile.name);
  const [useCase, setUseCase] = useState(profile.use_case);
  const [isDefault, setIsDefault] = useState(profile.is_default);
  const [loadSettings, setLoadSettings] = useState(profile.load_settings_json);
  const [requestSettings, setRequestSettings] = useState(profile.request_settings_json);
  const [visibility, setVisibility] = useState<Visibility>("basic");
  const refresh = () => void client.invalidateQueries({ queryKey: ["profiles"] });
  const save = useMutation({
    mutationFn: () => api.updateProfile(profile.id, {
      name,
      use_case: useCase,
      is_default: isDefault,
      load_settings: loadSettings,
      request_settings: requestSettings,
    }),
    onSuccess: () => { refresh(); onClose(); },
  });
  const clone = useMutation({
    mutationFn: () => api.cloneProfile(profile.id),
    onSuccess: () => { refresh(); onClose(); },
  });
  const reset = useMutation({
    mutationFn: () => api.resetProfile(profile.id),
    onSuccess: (value) => {
      setLoadSettings(value.load_settings_json);
      setRequestSettings(value.request_settings_json);
      refresh();
    },
  });
  const remove = useMutation({
    mutationFn: () => api.deleteProfile(profile.id),
    onSuccess: () => { refresh(); onClose(); },
  });
  const exportBundle = useMutation({
    mutationFn: () => api.exportProfile(profile.id),
    onSuccess: (bundle) => downloadJson(bundle, `${profile.name.replaceAll(/[^a-z0-9]+/gi, "-").toLowerCase()}.lm-atelier-profile.json`),
  });
  const engine = engines.find((item) => item.engine === profile.engine && item.roles.includes(profile.role))
    ?? engines.find((item) => item.roles.includes(profile.role));
  const fields = resolveCapabilitySettings(engine, profile.role).filter(
    (field) => visibilityRank[field.visibility] <= visibilityRank[visibility] && field.available,
  );
  const error = save.error ?? clone.error ?? reset.error ?? remove.error ?? exportBundle.error;
  return (
    <AccessibleDialog
      title="Edit profile"
      eyebrow={`${profile.role} profile · ${profile.engine}`}
      closeLabel="Close profile editor"
      onClose={onClose}
      className="settings-editor"
    >
      <label>Profile name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label>Best used for<textarea rows={3} value={useCase} onChange={(event) => setUseCase(event.target.value)} placeholder="Programming, code review, technical explanations" /></label>
      <label className="toggle-row"><span><strong>Default {profile.role} model</strong><small>Used by chats set to Default. Auto uses it when no use case matches.</small></span><input type="checkbox" checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} /></label>
      <div className="segmented compact" role="group" aria-label="Profile setting detail">
        {(["basic", "advanced", "expert"] as Visibility[]).map((level) => <button type="button" key={level} className={visibility === level ? "active" : ""} aria-pressed={visibility === level} onClick={() => setVisibility(level)}>{level}</button>)}
      </div>
      <div className="settings-list embedded">
        {fields.map((field) => {
          const target = field.scope === "load" ? loadSettings : requestSettings;
          return <div className="scoped-setting" key={`${field.scope}:${field.key}:${JSON.stringify(target[field.key])}`}><span className="scope-label">{field.scope}{field.restart_required ? " · restart required" : ""}</span><SettingControl field={field} value={target[field.key] ?? field.default} onChange={(value) => field.scope === "load" ? setLoadSettings({ ...loadSettings, [field.key]: value }) : setRequestSettings({ ...requestSettings, [field.key]: value })} /></div>;
        })}
        {!engine && <p className="muted">No capability schema is available for this profile engine.</p>}
      </div>
      {error && <ErrorCallout message={error.message} />}
      <footer className="editor-actions"><button className="secondary danger" onClick={() => remove.mutate()} disabled={remove.isPending}>Delete profile</button><button className="secondary" onClick={() => reset.mutate()} disabled={reset.isPending}>Reset settings</button><button className="secondary" onClick={() => exportBundle.mutate()}>Export</button><button className="secondary" onClick={() => clone.mutate()}>Clone</button><button className="primary" onClick={() => save.mutate()} disabled={!name.trim() || save.isPending}>Save profile</button></footer>
    </AccessibleDialog>
  );
}

function PresetEditor({
  preset,
  engines,
  onClose,
}: {
  preset: GenerationPreset;
  engines: EngineCapabilities[];
  onClose: () => void;
}) {
  const client = useQueryClient();
  const [name, setName] = useState(preset.name);
  const [isDefault, setIsDefault] = useState(preset.is_default);
  const [settings, setSettings] = useState(preset.settings_json);
  const [visibility, setVisibility] = useState<Visibility>("basic");
  const refresh = () => void client.invalidateQueries({ queryKey: ["presets"] });
  const save = useMutation({ mutationFn: () => api.updatePreset(preset.id, { name, is_default: isDefault, settings }), onSuccess: () => { refresh(); onClose(); } });
  const clone = useMutation({ mutationFn: () => api.clonePreset(preset.id), onSuccess: () => { refresh(); onClose(); } });
  const reset = useMutation({ mutationFn: () => api.resetPreset(preset.id), onSuccess: (value) => { setSettings(value.settings_json); refresh(); } });
  const remove = useMutation({ mutationFn: () => api.deletePreset(preset.id), onSuccess: () => { refresh(); onClose(); } });
  const exportBundle = useMutation({ mutationFn: () => api.exportPreset(preset.id), onSuccess: (bundle) => downloadJson(bundle, `${preset.name.replaceAll(/[^a-z0-9]+/gi, "-").toLowerCase()}.lm-atelier-preset.json`) });
  const engine = engines.find((item) => item.roles.includes(preset.role));
  const fields = resolveCapabilitySettings(engine, preset.role).filter((field) => field.scope !== "load" && visibilityRank[field.visibility] <= visibilityRank[visibility] && field.available);
  const error = save.error ?? clone.error ?? reset.error ?? remove.error ?? exportBundle.error;
  return (
    <AccessibleDialog
      title="Edit preset"
      eyebrow={`${preset.role} generation preset`}
      closeLabel="Close preset editor"
      onClose={onClose}
      className="settings-editor"
    >
      <label>Preset name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label className="toggle-row"><span><strong>Default {preset.role} preset</strong></span><input type="checkbox" checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} /></label>
      <div className="segmented compact" role="group" aria-label="Preset setting detail">{(["basic", "advanced", "expert"] as Visibility[]).map((level) => <button type="button" key={level} className={visibility === level ? "active" : ""} aria-pressed={visibility === level} onClick={() => setVisibility(level)}>{level}</button>)}</div>
      <div className="settings-list embedded">{fields.map((field) => <div className="scoped-setting" key={`${field.scope}:${field.key}:${JSON.stringify(settings[field.key])}`}><span className="scope-label">{field.scope}</span><SettingControl field={field} value={settings[field.key] ?? field.default} onChange={(value) => setSettings({ ...settings, [field.key]: value })} /></div>)}</div>
      {error && <ErrorCallout message={error.message} />}
      <footer className="editor-actions"><button className="secondary danger" onClick={() => remove.mutate()}>Delete</button><button className="secondary" onClick={() => reset.mutate()}>Reset</button><button className="secondary" onClick={() => exportBundle.mutate()}>Export</button><button className="secondary" onClick={() => clone.mutate()}>Clone</button><button className="primary" onClick={() => save.mutate()} disabled={!name.trim() || save.isPending}>Save preset</button></footer>
    </AccessibleDialog>
  );
}

function SettingsView({ engines }: { engines: EngineCapabilities[] }) {
  const client = useQueryClient();
  const [selectedProfile, setSelectedProfile] = useState<ModelProfile | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<GenerationPreset | null>(null);
  const [presetName, setPresetName] = useState("");
  const [presetRole, setPresetRole] = useState<GenerationPreset["role"]>("chat");
  const [importError, setImportError] = useState("");
  const [backupFeedback, setBackupFeedback] = useState<{
    kind: "success" | "error";
    message: string;
  } | null>(null);
  const profileImport = useRef<HTMLInputElement>(null);
  const presetImport = useRef<HTMLInputElement>(null);
  const system = useQuery({ queryKey: ["system"], queryFn: api.system });
  const about = useQuery({ queryKey: ["about"], queryFn: api.about });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const workers = useQuery({ queryKey: ["workers"], queryFn: api.workers, refetchInterval: 3_000 });
  const runtimes = useQuery({
    queryKey: ["runtimes"],
    queryFn: api.runtimes,
    refetchInterval: (query) => query.state.data?.some((runtime) => runtime.state === "installing") ? 2_000 : false,
  });
  const backups = useQuery({ queryKey: ["backups"], queryFn: api.backups });
  const refreshWorkers = () => void client.invalidateQueries({ queryKey: ["workers"] });
  const loadChat = useMutation({ mutationFn: api.loadChatWorker, onSettled: refreshWorkers });
  const startMedia = useMutation({ mutationFn: api.startMediaWorker, onSettled: refreshWorkers });
  const stopWorker = useMutation({ mutationFn: api.stopWorker, onSettled: refreshWorkers });
  const installRuntime = useMutation({
    mutationFn: (engine: RuntimeStatus["engine"]) => api.installRuntime(engine),
    onSuccess: (value: RuntimeStatus) => {
      client.setQueryData<RuntimeStatus[]>(["runtimes"], (current = []) =>
        current.map((item) => item.engine === value.engine ? value : item)
      );
    },
  });
  const storeBackup = (backup: BackupInfo) => {
    client.setQueryData<BackupInfo[]>(["backups"], (current = []) => {
      const existing = current.find((item) => item.name === backup.name);
      const stored = backup.restore_pending
        ? backup
        : { ...backup, restore_pending: existing?.restore_pending ?? false };
      const remaining = current
        .filter((item) => item.name !== backup.name)
        .map((item) => backup.restore_pending ? { ...item, restore_pending: false } : item);
      return [stored, ...remaining].sort(
        (left, right) => Date.parse(right.created_at) - Date.parse(left.created_at),
      );
    });
  };
  const showBackupError = (error: Error) =>
    setBackupFeedback({ kind: "error", message: error.message });
  const createBackup = useMutation({
    mutationFn: api.createBackup,
    onMutate: () => setBackupFeedback(null),
    onSuccess: (backup) => {
      storeBackup(backup);
      setBackupFeedback({ kind: "success", message: "Backup created." });
    },
    onError: showBackupError,
  });
  const verifyBackup = useMutation({
    mutationFn: api.verifyBackup,
    onMutate: () => setBackupFeedback(null),
    onSuccess: (backup) => {
      storeBackup(backup);
      setBackupFeedback({ kind: "success", message: "Backup verified." });
    },
    onError: showBackupError,
  });
  const restoreBackup = useMutation({
    mutationFn: api.restoreBackup,
    onMutate: () => setBackupFeedback(null),
    onSuccess: (backup) => {
      storeBackup(backup);
      setBackupFeedback({
        kind: "success",
        message: "Restore scheduled. Restart LM Atelier to apply this backup.",
      });
    },
    onError: showBackupError,
  });
  const deleteBackup = useMutation({
    mutationFn: api.deleteBackup,
    onMutate: () => setBackupFeedback(null),
    onSuccess: (_value, name) => {
      client.setQueryData<BackupInfo[]>(["backups"], (current = []) =>
        current.filter((backup) => backup.name !== name)
      );
      setBackupFeedback({ kind: "success", message: "Backup deleted." });
    },
    onError: showBackupError,
  });
  const toolProbe = useMutation({ mutationFn: api.probeChatTools });
  const chatWorker = workers.data?.find((worker) => worker.name === "chat");
  const chatWorkerBusy = Boolean(
    chatWorker && chatWorker.active_jobs + chatWorker.queued_jobs > 0
  );
  const createPreset = useMutation({
    mutationFn: () => api.createPreset(presetRole, presetName),
    onSuccess: () => {
      setPresetName("");
      void client.invalidateQueries({ queryKey: ["presets"] });
    },
  });
  const setDefaultProfile = useMutation({
    mutationFn: (profile: ModelProfile) =>
      api.updateProfile(profile.id, { is_default: true }),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["profiles"] }),
  });
  const importBundle = async (file: File | undefined, kind: "profile" | "preset") => {
    if (!file) return;
    setImportError("");
    try {
      const value = JSON.parse(await file.text()) as ModelProfileBundle | GenerationPresetBundle;
      if (kind === "profile") {
        if (value.format !== "lm-atelier-profile") throw new Error("This is not an LM Atelier profile bundle.");
        await api.importProfile(value as ModelProfileBundle);
        await client.invalidateQueries({ queryKey: ["profiles"] });
      } else {
        if (value.format !== "lm-atelier-preset") throw new Error("This is not an LM Atelier preset bundle.");
        await api.importPreset(value as GenerationPresetBundle);
        await client.invalidateQueries({ queryKey: ["presets"] });
      }
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Could not import the bundle.");
    }
  };
  const technicalDetails = about.data && system.data
    ? formatTechnicalDetails(about.data, system.data, engines)
    : "";
  return (
    <div className="page-view settings-page">
      <header className="page-header"><div><h1>Settings</h1></div></header>
      <CredentialSettingsCard
        provider="huggingface"
        providerLabel="Hugging Face"
        description="Access private and gated models. The token stays in your operating-system credential vault and is never displayed after saving."
        environmentVariable="LOCAL_LM_HF_TOKEN"
        placeholder="hf_..."
      />
      <CredentialSettingsCard
        provider="civitai"
        providerLabel="CivitAI"
        description="Store a CivitAI API token securely for authenticated catalog downloads."
        environmentVariable="LOCAL_LM_CIVITAI_TOKEN"
        placeholder="CivitAI API token"
      />
      <section><h2>Engines</h2><div className="engine-grid">{engines.map((engine) => <article className="engine-card" key={`${engine.engine}:${engine.roles.join()}`}><header><div className="model-icon"><Cpu /></div><div><h3>{engine.engine}</h3><p>{engine.roles.join(" · ")} · {engine.version}</p></div><StatusDot healthy={engine.healthy} label={`${engine.engine} engine`} /></header>{engine.roles.includes("chat") && <div className="capability-list"><button className="secondary compact-button" onClick={() => toolProbe.mutate()} disabled={toolProbe.isPending}>{toolProbe.isPending ? "Testing…" : "Test structured tools"}</button></div>}</article>)}</div>{toolProbe.data && <div className={`callout ${toolProbe.data.passed ? "success" : "error"}`} role={toolProbe.data.passed ? "status" : "alert"}>{toolProbe.data.passed ? `Structured tool schema passed on ${toolProbe.data.engine} ${toolProbe.data.version}.` : `Structured tool schema failed: ${toolProbe.data.error || "unknown response"}`}</div>}{toolProbe.error && <ErrorCallout message={toolProbe.error.message} />}<div className="runtime-setup-grid">{runtimes.data?.map((runtime) => <RuntimeSetupCard key={runtime.engine} runtime={runtime} installPending={installRuntime.isPending} onInstall={(engine) => installRuntime.mutate(engine)} />)}</div>{(runtimes.error || installRuntime.error) && <ErrorCallout message={(runtimes.error || installRuntime.error)?.message} />}</section>
      <section><h2>Machine</h2>{system.data && <div className="metric-grid"><div className="cpu-metric"><Cpu /><span><strong>{system.data.cpu_model}</strong><small>CPU model</small></span></div><div><HardDrive /><span><strong>{formatBytes(system.data.disk_free_bytes)}</strong> disk free</span></div></div>}<div className="device-list">{system.data?.devices.filter((device) => device.kind !== "cpu").map((device) => <div key={device.id}><span className="device-icon"><Cpu size={18} /></span><span><strong>{device.name}</strong><small>{device.backend}</small></span></div>)}</div></section>
      <section>
        <div className="detail-title"><div><h2>Model profiles</h2></div><button className="secondary" onClick={() => profileImport.current?.click()}>Import profile</button></div>
        <input ref={profileImport} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0], "profile"); event.target.value = ""; }} />
        <div className="profile-table interactive">{profiles.data?.map((profile: ModelProfile) => <div key={profile.id}><span className="badge">{profile.role}</span><strong>{profile.name}{profile.is_default ? " · default" : ""}</strong><span title={profile.use_case}>{profile.use_case || "No Auto use case yet"}</span><span className="row-actions">{!profile.is_default && <button className="secondary compact-button" aria-label={`Set ${profile.name} as default ${profile.role} model`} disabled={setDefaultProfile.isPending} onClick={() => setDefaultProfile.mutate(profile)}>Set default</button>}{profile.role === "chat" && profile.model_install_id && <button className="secondary compact-button" aria-label={`Load profile: ${profile.name}`} disabled={chatWorkerBusy || loadChat.isPending} title={chatWorkerBusy ? "Wait for active and queued jobs before changing the worker" : "Load this chat profile"} onClick={() => loadChat.mutate(profile.id)}>Load</button>}<button className="secondary compact-button" aria-label={`Edit profile: ${profile.name}`} onClick={() => setSelectedProfile(profile)}>Edit</button></span></div>)}</div>
        {setDefaultProfile.error && <ErrorCallout message={setDefaultProfile.error.message} />}
      </section>
      <section>
        <div className="detail-title"><div><h2>Generation presets</h2><p>Reuse response length, sampling, image size, video length, and seed settings.</p></div><button className="secondary" onClick={() => presetImport.current?.click()}>Import preset</button></div>
        <input ref={presetImport} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0], "preset"); event.target.value = ""; }} />
        <div className="preset-create"><input aria-label="New preset name" placeholder="New preset name" value={presetName} onChange={(event) => setPresetName(event.target.value)} /><select aria-label="New preset role" value={presetRole} onChange={(event) => setPresetRole(event.target.value as GenerationPreset["role"])}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option></select><button className="primary" disabled={!presetName.trim() || createPreset.isPending} onClick={() => createPreset.mutate()}><Plus size={15} />Create preset</button></div>
        <div className="profile-table interactive">{presets.data?.map((preset) => <div key={preset.id}><span className="badge">{preset.role}</span><strong>{preset.name}{preset.is_default ? " · default" : ""}</strong><span>{Object.keys(preset.settings_json).length} overrides</span><button className="secondary compact-button" aria-label={`Edit preset: ${preset.name}`} onClick={() => setSelectedPreset(preset)}>Edit</button></div>)}</div>
        {(createPreset.error || importError) && <ErrorCallout message={createPreset.error?.message || importError} />}
      </section>
      <section>
        <div className="detail-title">
          <div><h2>Workers</h2><p>A model must finish loading within the startup time limit. Large models on slow disks can need more than the default 60 seconds.</p></div>
          <div className="row-actions">
            <WorkerLogFolderButton />
            <WorkerStartupLimit />
          </div>
        </div>
        <div className="engine-grid">
          {workers.data?.map((worker) => (
            <WorkerStatusCard
              key={worker.name}
              worker={worker}
              startPending={startMedia.isPending}
              stopPending={stopWorker.isPending}
              onStart={() => startMedia.mutate()}
              onStop={() => stopWorker.mutate(worker.name)}
            />
          ))}
        </div>
        <ErrorCallout
          message={(loadChat.error || startMedia.error || stopWorker.error)?.message}
        />
      </section>
      <section>
        <div className="detail-title">
          <div><h2>Recovery backups</h2></div>
          <div className="row-actions">
            <DownloadDiagnosticsButton />
            <button
              className="secondary"
              disabled={createBackup.isPending}
              onClick={() => createBackup.mutate(false)}
            >
              {createBackup.isPending && createBackup.variables === false ? "Backing up…" : "Back up state"}
            </button>
            <button
              className="secondary"
              disabled={createBackup.isPending}
              onClick={() => createBackup.mutate(true)}
            >
              {createBackup.isPending && createBackup.variables === true ? "Backing up…" : "Back up with media"}
            </button>
          </div>
        </div>
        {backups.data?.some((backup) => backup.restore_pending) && (
          <div className="callout success" role="status">
            Restore scheduled. Restart LM Atelier to apply the selected backup.
          </div>
        )}
        {backups.data?.length ? (
          <div className="backup-list">
            {backups.data.map((backup) => {
              const verifying = verifyBackup.isPending && verifyBackup.variables === backup.name;
              const restoring = restoreBackup.isPending && restoreBackup.variables === backup.name;
              const deleting = deleteBackup.isPending && deleteBackup.variables === backup.name;
              return (
                <article className="backup-row" key={backup.name}>
                  <div className="backup-copy">
                    <strong>{formatDate(backup.created_at)}</strong>
                    <small>
                      {formatBytes(backup.size_bytes + backup.media_size_bytes)}
                      {" · "}
                      {backup.media_included ? "State + media" : "State only"}
                      {" · "}
                      {backup.verified ? "Verified" : "Not verified"}
                    </small>
                    <code title={backup.name}>{backup.name}</code>
                  </div>
                  <div className="row-actions">
                    <button
                      className="secondary compact-button"
                      aria-label={`Verify backup ${backup.name}`}
                      disabled={verifying || restoring || deleting}
                      onClick={() => verifyBackup.mutate(backup.name)}
                    >
                      {verifying ? "Verifying…" : "Verify"}
                    </button>
                    <button
                      className="secondary compact-button"
                      aria-label={`Restore backup ${backup.name} on restart`}
                      disabled={backup.restore_pending || verifying || restoring || deleting}
                      onClick={() => {
                        if (window.confirm("Restore this backup the next time LM Atelier starts?")) {
                          restoreBackup.mutate(backup.name);
                        }
                      }}
                    >
                      {restoring ? "Scheduling…" : backup.restore_pending ? "Restore scheduled" : "Restore on restart"}
                    </button>
                    <button
                      className="secondary compact-button danger"
                      aria-label={`Delete backup ${backup.name}`}
                      disabled={backup.restore_pending || verifying || restoring || deleting}
                      onClick={() => {
                        if (window.confirm("Delete this recovery backup? This cannot be undone.")) {
                          deleteBackup.mutate(backup.name);
                        }
                      }}
                    >
                      {deleting ? "Deleting…" : "Delete"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          !backups.isLoading && <p className="muted">No recovery backups yet.</p>
        )}
        {backupFeedback && (
          <div
            className={`callout ${backupFeedback.kind}`}
            role={backupFeedback.kind === "error" ? "alert" : "status"}
          >
            {backupFeedback.message}
          </div>
        )}
        {backups.error && <ErrorCallout message={backups.error.message} />}
      </section>
      <section>
        <div className="detail-title">
          <div><h2>About &amp; support</h2></div>
          {about.data && <span className="badge">Version {about.data.version}</span>}
        </div>
        {about.data && <div className="about-support">
          <div className="about-paths">
            <div><Folder size={17} /><span><small>Data folder</small><code>{about.data.data_directory}</code></span><CopyTextButton text={about.data.data_directory} label="Copy data folder" buttonText="Copy data folder" className="secondary compact-button" /></div>
            <div><Folder size={17} /><span><small>Log folder</small><code>{about.data.log_directory}</code></span><CopyTextButton text={about.data.log_directory} label="Copy log folder" buttonText="Copy log folder" className="secondary compact-button" /></div>
          </div>
          <div className="about-actions">
            {technicalDetails && <CopyTextButton text={technicalDetails} label="Copy technical details" buttonText="Copy technical details" className="secondary" />}
            <nav aria-label="Support resources">
              {supportLinks(about.data.version).map(([label, href]) => (
                <a key={label} href={href} target="_blank" rel="noreferrer">{label}</a>
              ))}
            </nav>
          </div>
        </div>}
        {(about.error || system.error) && <ErrorCallout message="About information is unavailable." />}
      </section>
      {selectedProfile && <ProfileEditor profile={selectedProfile} engines={engines} onClose={() => setSelectedProfile(null)} />}
      {selectedPreset && <PresetEditor preset={selectedPreset} engines={engines} onClose={() => setSelectedPreset(null)} />}
    </div>
  );
}

function ChatManager({
  chat,
  projects,
  onClose,
  onSave,
  onDelete,
}: {
  chat: Chat;
  projects: Project[];
  onClose: () => void;
  onSave: (values: Partial<Chat>) => void;
  onDelete: (deleteGeneratedMedia: boolean) => void;
}) {
  const [title, setTitle] = useState(chat.title);
  const [projectId, setProjectId] = useState(chat.project_id ?? "");
  const [archived, setArchived] = useState(chat.archived);
  const [confirmUncertainMedia, setConfirmUncertainMedia] = useState(chat.confirm_uncertain_media);
  const [verifyImageEdits, setVerifyImageEdits] = useState(
    chat.vision_settings_json?.verify_image_edits === true,
  );
  const [compileVisualPrompts, setCompileVisualPrompts] = useState(
    chat.vision_settings_json?.compile_visual_prompts !== false,
  );
  const [deleteGeneratedMedia, setDeleteGeneratedMedia] = useState(false);
  const deletePrompt = deleteGeneratedMedia
    ? `Delete ${chat.title}, its history, and generated media used only by this chat?`
    : `Delete ${chat.title} and its history?`;
  return (
    <AccessibleDialog
      title="Manage chat"
      eyebrow="Conversation"
      closeLabel="Close chat manager"
      onClose={onClose}
      className="workspace-editor"
    >
      <label>Title<input value={title} onChange={(event) => setTitle(event.target.value)} /></label>
      <label>Project<select value={projectId} onChange={(event) => setProjectId(event.target.value)}><option value="">Unfiled</option>{projects.filter((project) => !project.archived).map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></label>
      <label className="toggle-row"><span className="toggle-copy"><strong>Confirm uncertain media</strong><small>Ask before Auto mode starts an image or video when the planner is unsure.</small></span><input type="checkbox" checked={confirmUncertainMedia} onChange={(event) => setConfirmUncertainMedia(event.target.checked)} /></label>
      <label className="toggle-row"><span className="toggle-copy"><strong>Review image edits</strong><small>Check the result locally and retry once when the requested change is missing.</small></span><input type="checkbox" checked={verifyImageEdits} onChange={(event) => setVerifyImageEdits(event.target.checked)} /></label>
      <label className="toggle-row"><span className="toggle-copy"><strong>Compose visual prompts</strong><small>When a request asks to picture something written earlier, rewrite that passage as one scene description before generating.</small></span><input type="checkbox" checked={compileVisualPrompts} onChange={(event) => setCompileVisualPrompts(event.target.checked)} /></label>
      <label className="toggle-row"><span className="toggle-copy"><strong>Archived</strong><small>Hide this chat from the active workspace without deleting its history.</small></span><input type="checkbox" checked={archived} onChange={(event) => setArchived(event.target.checked)} /></label>
      <label className="toggle-row delete-media-option"><span className="toggle-copy"><strong>Delete generated media with chat</strong><small>Permanently delete image and video outputs used only by this chat. Shared media is kept.</small></span><input type="checkbox" checked={deleteGeneratedMedia} onChange={(event) => setDeleteGeneratedMedia(event.target.checked)} /></label>
      <footer className="editor-actions"><button className="secondary danger" onClick={() => { if (window.confirm(deletePrompt)) onDelete(deleteGeneratedMedia); }}>Delete chat</button><button className="secondary" onClick={onClose}>Cancel</button><button className="primary" disabled={!title.trim()} onClick={() => onSave({ title: title.trim(), project_id: projectId || null, archived, confirm_uncertain_media: confirmUncertainMedia, vision_settings_json: { ...(chat.vision_settings_json ?? {}), verify_image_edits: verifyImageEdits, compile_visual_prompts: compileVisualPrompts } })}>Save chat</button></footer>
    </AccessibleDialog>
  );
}

function ProjectManager({
  project,
  engines,
  presets,
  workflows,
  onClose,
  onSave,
  onDelete,
  onExport,
}: {
  project: Project;
  engines: EngineCapabilities[];
  presets: GenerationPreset[];
  workflows: Workflow[];
  onClose: () => void;
  onSave: (values: Partial<Project>) => void;
  onDelete: () => void;
  onExport: (includeMedia: boolean) => void;
}) {
  const [name, setName] = useState(project.name);
  const [description, setDescription] = useState(project.description);
  const [instructions, setInstructions] = useState(project.instructions);
  const [archived, setArchived] = useState(project.archived);
  const [imageWorkflowRevisionId, setImageWorkflowRevisionId] = useState(project.image_workflow_revision_id ?? "");
  const [videoWorkflowRevisionId, setVideoWorkflowRevisionId] = useState(project.video_workflow_revision_id ?? "");
  const [settingsRole, setSettingsRole] = useState<EngineRole>("chat");
  const [generationSettings, setGenerationSettings] = useState<
    NonNullable<Project["generation_settings_json"]>
  >(() => Object.fromEntries(
    Object.entries(project.generation_settings_json ?? {}).map(([role, settings]) => [
      role,
      { ...settings },
    ]),
  ));
  const [generationPresetIds, setGenerationPresetIds] = useState<
    NonNullable<Project["generation_preset_ids_json"]>
  >({ ...(project.generation_preset_ids_json ?? {}) });
  const workflowOptions = (kind: "image" | "video") => workflows
    .filter((workflow) => workflow.operation.includes(kind))
    .flatMap((workflow) => workflow.revisions.map((revision) => (
      <option key={revision.id} value={revision.id}>
        {workflow.name} · {workflow.operation.replaceAll("_", " ")} · v{revision.version}
      </option>
    )));
  const setRoleSettings = (values: Record<string, unknown>) => {
    setGenerationSettings((current) => {
      const next = { ...current };
      if (Object.keys(values).length) next[settingsRole] = values;
      else delete next[settingsRole];
      return next;
    });
  };
  const setRolePreset = (presetId: string | null) => {
    setGenerationPresetIds((current) => {
      const next = { ...current };
      if (presetId) next[settingsRole] = presetId;
      else delete next[settingsRole];
      return next;
    });
  };
  const clearRoleDefaults = () => {
    setRoleSettings({});
    setRolePreset(null);
  };
  const clearAllDefaults = () => {
    setGenerationSettings({});
    setGenerationPresetIds({});
  };
  return (
    <AccessibleDialog
      title="Manage project"
      eyebrow="Workspace"
      closeLabel="Close project manager"
      onClose={onClose}
      className="workspace-editor project-editor"
    >
      <label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label>Description<textarea rows={3} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
      <label>Project instructions<textarea rows={5} value={instructions} onChange={(event) => setInstructions(event.target.value)} /></label>
      <label>Default image workflow revision<select value={imageWorkflowRevisionId} onChange={(event) => setImageWorkflowRevisionId(event.target.value)}><option value="">Use global default</option>{workflowOptions("image")}</select></label>
      <label>Default video workflow revision<select value={videoWorkflowRevisionId} onChange={(event) => setVideoWorkflowRevisionId(event.target.value)}><option value="">Use global default</option>{workflowOptions("video")}</select></label>
      <section className="project-generation-defaults" aria-labelledby="project-generation-defaults-heading">
        <div className="project-defaults-heading">
          <div>
            <h3 id="project-generation-defaults-heading">Generation defaults</h3>
            <p>Chats inherit these settings; chat and turn choices override them.</p>
          </div>
          <button className="secondary compact-button" type="button" onClick={clearAllDefaults}>Clear all</button>
        </div>
        <div className="segmented project-role-tabs" aria-label="Project generation role">
          {(["chat", "image", "video"] as EngineRole[]).map((role) => (
            <button
              key={role}
              type="button"
              className={settingsRole === role ? "active" : ""}
              aria-pressed={settingsRole === role}
              onClick={() => setSettingsRole(role)}
            >
              {role}
            </button>
          ))}
        </div>
        <GenerationSettingsPanel
          key={settingsRole}
          role={settingsRole}
          engines={engines}
          values={generationSettings[settingsRole] ?? {}}
          onValues={setRoleSettings}
          presets={presets}
          presetId={generationPresetIds[settingsRole] ?? null}
          onPreset={setRolePreset}
          presetLabel={`${settingsRole} project preset`}
          resetLabel="Clear role defaults"
          onReset={clearRoleDefaults}
        />
      </section>
      <label className="toggle-row"><span><strong>Archived</strong><small>Hide this project while preserving its chats and media.</small></span><input type="checkbox" checked={archived} onChange={(event) => setArchived(event.target.checked)} /></label>
      <div className="project-export-actions"><button className="secondary" onClick={() => onExport(false)}>Export metadata only</button><button className="secondary" onClick={() => onExport(true)}>Export with media</button></div>
      <footer className="editor-actions">
        <button className="secondary danger" onClick={() => { if (window.confirm(`Delete ${project.name}? Its chats will become unfiled.`)) onDelete(); }}>Delete project</button>
        <button className="secondary" onClick={onClose}>Cancel</button>
        <button
          className="primary"
          disabled={!name.trim()}
          onClick={() => onSave({
            name: name.trim(),
            description,
            instructions,
            archived,
            image_workflow_revision_id: imageWorkflowRevisionId || null,
            video_workflow_revision_id: videoWorkflowRevisionId || null,
            generation_settings_json: generationSettings,
            generation_preset_ids_json: generationPresetIds,
          })}
        >
          Save project
        </button>
      </footer>
    </AccessibleDialog>
  );
}

function Sidebar({
  projects,
  chats,
  engines,
  presets,
  workflows,
  currentChatId,
  view,
  setupState,
  onChat,
  onSetup,
  onView,
  onNewChat,
  onNewProject,
  onExportProject,
  onImportProject,
  onUpdateChat,
  onDeleteChat,
  onUpdateProject,
  onDeleteProject,
}: {
  projects: Project[];
  chats: Chat[];
  engines: EngineCapabilities[];
  presets: GenerationPreset[];
  workflows: Workflow[];
  currentChatId: string | null;
  view: View;
  setupState?: SetupReadinessReport["state"] | undefined;
  onChat: (id: string) => void;
  onSetup: () => void;
  onView: (view: View) => void;
  onNewChat: (projectId?: string | null) => void;
  onNewProject: () => void;
  onExportProject: (id: string, includeMedia?: boolean) => void;
  onImportProject: (file: File) => void;
  onUpdateChat: (id: string, values: Partial<Chat>) => void;
  onDeleteChat: (id: string, deleteGeneratedMedia: boolean) => void;
  onUpdateProject: (id: string, values: Partial<Project>) => void;
  onDeleteProject: (id: string) => void;
}) {
  const [closedProjects, setClosedProjects] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [managedChat, setManagedChat] = useState<Chat | null>(null);
  const [managedProject, setManagedProject] = useState<Project | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const projectImport = useRef<HTMLInputElement>(null);
  const normalizedSearch = search.trim().toLowerCase();
  const visibleChats = chats.filter((chat) => (showArchived || !chat.archived) && (!normalizedSearch || chat.title.toLowerCase().includes(normalizedSearch)));
  const visibleProjects = projects.filter((project) => (showArchived || !project.archived) && (!normalizedSearch || project.name.toLowerCase().includes(normalizedSearch) || visibleChats.some((chat) => chat.project_id === project.id)));
  const unfiled = visibleChats.filter((chat) => !chat.project_id);
  const chatRow = (chat: Chat) => <div className="sidebar-chat-row" key={chat.id}><button className={`chat-main ${view === "chat" && currentChatId === chat.id ? "active" : ""}`} aria-current={view === "chat" && currentChatId === chat.id ? "page" : undefined} onClick={() => { onChat(chat.id); setMobileOpen(false); }}><MessageSquare size={14} /><span>{chat.title}</span>{chat.archived && <small>Archived</small>}</button><button className="inline-add" aria-label={`Manage ${chat.title}`} onClick={() => setManagedChat(chat)}><MoreHorizontal size={13} /></button></div>;
  return (
    <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`}>
      <div className="brand"><div className="brand-mark"><AtelierMark /></div><span>LM Atelier<small>Local creative studio</small></span><button className="icon-button mobile-menu" aria-label="Toggle navigation" aria-expanded={mobileOpen} onClick={() => setMobileOpen((open) => !open)}><Menu /></button></div>
      <button className="new-chat" onClick={() => { onNewChat(null); setMobileOpen(false); }}><Plus size={18} />New chat</button>
      <nav className="primary-nav"><button className={view === "media" ? "active" : ""} aria-current={view === "media" ? "page" : undefined} onClick={() => { onView("media"); setMobileOpen(false); }}><ImageIcon />Media library</button><button className={view === "models" ? "active" : ""} aria-current={view === "models" ? "page" : undefined} onClick={() => { onView("models"); setMobileOpen(false); }}><Library />Model library</button><button className={view === "workflows" ? "active" : ""} aria-current={view === "workflows" ? "page" : undefined} onClick={() => { onView("workflows"); setMobileOpen(false); }}><WorkflowIcon />Workflows</button></nav>
      <div className="workspace-search"><Search size={14} /><input aria-label="Search projects and chats" placeholder="Search workspace" value={search} onChange={(event) => setSearch(event.target.value)} /><button className={showArchived ? "active" : ""} aria-pressed={showArchived} onClick={() => setShowArchived((value) => !value)}>Archived</button></div>
      <div className="workspace-tree" role="region" aria-label="Projects and chats">
        <div className="sidebar-section">
          <div className="section-title"><span>Projects</span><input ref={projectImport} hidden type="file" accept=".zip,.lm-atelier.zip,application/zip" onChange={(event) => { const file = event.target.files?.[0]; if (file) onImportProject(file); event.target.value = ""; }} /><button aria-label="Import project" onClick={() => projectImport.current?.click()}><Upload size={14} /></button><button aria-label="New project" onClick={onNewProject}><Plus size={15} /></button></div>
          {visibleProjects.map((project) => {
            const open = !closedProjects.has(project.id);
            const projectMatches = normalizedSearch && project.name.toLowerCase().includes(normalizedSearch);
            const projectChats = chats.filter((chat) => chat.project_id === project.id && (showArchived || !chat.archived) && (!normalizedSearch || projectMatches || chat.title.toLowerCase().includes(normalizedSearch)));
            return (
              <div className="project-group" key={project.id}>
                <div className="project-row">
                  <button className="project-main" aria-expanded={open} onClick={() => setClosedProjects((current) => {
                    const next = new Set(current);
                    if (open) next.add(project.id);
                    else next.delete(project.id);
                    return next;
                  })}>
                    <ChevronDown className={open ? "" : "closed"} size={14} />
                    <Folder size={16} />
                    <span>{project.name}</span>
                  </button>
                  <button className="inline-add" onClick={() => { onNewChat(project.id); setMobileOpen(false); }} aria-label={`New chat in ${project.name}`}><Plus size={13} /></button>
                  <button className="inline-add" onClick={() => onExportProject(project.id)} aria-label={`Export ${project.name}`}><Download size={13} /></button>
                  <button className="inline-add" onClick={() => setManagedProject(project)} aria-label={`Manage ${project.name}`}><MoreHorizontal size={13} /></button>
                </div>
                {open && <div className="chat-list">{projectChats.map(chatRow)}</div>}
              </div>
            );
          })}
        </div>
        {unfiled.length > 0 && <div className="sidebar-section"><div className="section-title"><span>Chats</span></div><div className="chat-list standalone">{unfiled.map(chatRow)}</div></div>}
      </div>
      <div className="sidebar-footer">
        <button onClick={() => { onSetup(); setMobileOpen(false); }}>
          <Sparkles />Setup
          {setupState && (
            <small className={`setup-nav-state ${setupState}`}>
              {setupState === "ready" ? "Ready" : setupState === "in_progress" ? "Working" : "Action needed"}
            </small>
          )}
        </button>
        <button className={view === "settings" ? "active" : ""} aria-current={view === "settings" ? "page" : undefined} onClick={() => { onView("settings"); setMobileOpen(false); }}><Settings />Settings</button>
      </div>
      {managedChat && <ChatManager chat={managedChat} projects={projects} onClose={() => setManagedChat(null)} onSave={(values) => { onUpdateChat(managedChat.id, values); setManagedChat(null); }} onDelete={(deleteGeneratedMedia) => { onDeleteChat(managedChat.id, deleteGeneratedMedia); setManagedChat(null); }} />}
      {managedProject && <ProjectManager project={managedProject} engines={engines} presets={presets} workflows={workflows} onClose={() => setManagedProject(null)} onSave={(values) => { onUpdateProject(managedProject.id, values); setManagedProject(null); }} onDelete={() => { onDeleteProject(managedProject.id); setManagedProject(null); }} onExport={(includeMedia) => onExportProject(managedProject.id, includeMedia)} />}
    </aside>
  );
}


export default function App() {
  const client = useQueryClient();
  const [view, setView] = useState<View>("chat");
  const [libraryEdit, setLibraryEdit] = useState<VisualTarget | null>(null);
  const [modelLibraryRole, setModelLibraryRole] = useState<EngineRole>("chat");
  const [setupOpen, setSetupOpen] = useState<boolean | null>(null);
  const [currentChatId, setCurrentChatId] = useState<string | null>(() => localStorage.getItem(CURRENT_CHAT_KEY));
  const [liveText, setLiveText] = useState<Record<string, string>>({});
  const [chatDrafts, setChatDrafts] = useState<Record<string, Partial<Chat>>>({});
  const [pendingTurns, setPendingTurns] = useState<Record<string, PendingTurn[]>>({});
  const setupReadiness = useQuery({
    queryKey: ["setup-readiness"],
    queryFn: api.setupReadiness,
    refetchInterval: (query) => query.state.data?.state === "ready" ? false : 3_000,
  });
  const setupVisible = setupOpen ?? Boolean(setupReadiness.data && setupReadiness.data.state !== "ready" && sessionStorage.getItem(SETUP_DISMISSED_KEY) !== "1");
  const [firstRunSetup, exitFirstRunSetup] = useFirstRunSetup();
  const projects = useQuery({
    queryKey: ["projects"],
    queryFn: () => api.projects(true),
  });
  const chats = useQuery({ queryKey: ["chats"], queryFn: () => api.chats(null, true) });
  const firstActiveChatId = chats.data?.find((candidate) => !candidate.archived)?.id ?? null;
  const activeChatId = currentChatId ?? firstActiveChatId;
  const chat = useQuery({ queryKey: ["chat", activeChatId], queryFn: () => api.chat(activeChatId!), enabled: Boolean(activeChatId) });
  const workPlans = useQuery({
    queryKey: ["work-plans", activeChatId],
    queryFn: () => api.workPlans(activeChatId!),
    enabled: Boolean(activeChatId),
    refetchInterval: 3_000,
  });
  const engines = useQuery({ queryKey: ["engines"], queryFn: api.engines });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const workflows = useQuery({ queryKey: ["workflows"], queryFn: api.workflows });
  const eventsConnected = useLiveEvents(client, setLiveText);

  const createChat = useMutation({
    mutationFn: (projectId?: string | null) => api.createChat(projectId),
    onSuccess: (created) => {
      setCurrentChatId(created.id);
      localStorage.setItem(CURRENT_CHAT_KEY, created.id);
      setView("chat");
      focusMainContent();
      void client.invalidateQueries({ queryKey: ["chats"] });
    },
  });
  const createProject = useMutation({
    mutationFn: api.createProject,
    onSuccess: () => void client.invalidateQueries({ queryKey: ["projects"] }),
  });
  const applyAcceptedTurn = (chatId: string, accepted: TurnAccepted) => {
    client.setQueryData<ChatDetail>(["chat", chatId], (current) => {
      if (!current) return current;
      const messageIds = new Set(current.messages.map((message) => message.id));
      const acceptedMessages = [accepted.user_message, accepted.assistant_message]
        .filter((message) => !messageIds.has(message.id));
      return {
        ...current,
        active_head_message_id: accepted.assistant_message.id,
        messages: [...current.messages, ...acceptedMessages],
      };
    });
    void client.invalidateQueries({ queryKey: ["chat", chatId], exact: true });
    void client.invalidateQueries({ queryKey: ["chats"] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
    void client.invalidateQueries({ queryKey: ["work-plans", chatId] });
  };
  const send = useMutation({
    mutationFn: ({ chatId, id, text, mode, artifacts, settings, stopCurrent }: SendTurnVariables) =>
      stopCurrent
        ? api.stopAndSendTurn(chatId, text, mode, artifacts, settings, id)
        : api.sendTurn(chatId, text, mode, artifacts, settings, id),
    onMutate: ({ chatId, id, text, mode }) => {
      setPendingTurns((current) => ({
        ...current,
        [chatId]: [...(current[chatId] ?? []), { id, text, mode }],
      }));
    },
    onSuccess: (accepted, { chatId }) => applyAcceptedTurn(chatId, accepted),
    onSettled: (_accepted, _error, { chatId, id }) => {
      setPendingTurns((current) => {
        const remaining = (current[chatId] ?? []).filter((pending) => pending.id !== id);
        const next = { ...current };
        if (remaining.length) next[chatId] = remaining;
        else delete next[chatId];
        return next;
      });
    },
  });
  const cancelWorkPlan = useMutation({
    mutationFn: api.cancelWorkPlan,
    onSuccess: (plan) => {
      void client.invalidateQueries({ queryKey: ["chat", plan.chat_id] });
      void client.invalidateQueries({ queryKey: ["work-plans", plan.chat_id] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const { deleteExchange, forkThread } = useMessageActions(setCurrentChatId, setView);
  const refreshWorkStep = () => {
    void client.invalidateQueries({ queryKey: ["chat", activeChatId] });
    void client.invalidateQueries({ queryKey: ["work-plans", activeChatId] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
  };
  const cancelWorkStep = useMutation({
    mutationFn: api.cancelWorkStep,
    onSuccess: refreshWorkStep,
  });
  const retryWorkStep = useMutation({
    mutationFn: api.retryWorkStep,
    onSuccess: refreshWorkStep,
  });
  const regenerate = useMutation({
    mutationFn: ({ messageId, settings }: { chatId: string; messageId: string; settings: Record<string, unknown> }) =>
      api.regenerateMessage(messageId, settings),
    onSuccess: (_accepted, { chatId }) => {
      void client.invalidateQueries({ queryKey: ["chat", chatId], exact: true });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const selectResponseRevision = useMutation({
    mutationFn: ({ messageId, revisionId }: { chatId: string; messageId: string; revisionId: string }) =>
      api.selectResponseRevision(messageId, revisionId),
    onSuccess: (_message, { chatId }) => {
      void client.invalidateQueries({ queryKey: ["chat", chatId], exact: true });
    },
  });
  const branch = useMutation({
    mutationFn: ({ messageId, text, mode, settings }: { chatId: string; messageId: string; text: string; mode: RoutingMode; settings: Record<string, unknown> }) =>
      api.branchMessage(messageId, text, mode, settings),
    onSuccess: (accepted, { chatId }) => applyAcceptedTurn(chatId, accepted),
  });
  const stop = useMutation({
    mutationFn: (chatId: string) => api.cancelChat(chatId),
    onSuccess: (_job, chatId) => {
      void client.invalidateQueries({ queryKey: ["chat", chatId] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const updateChat = useMutation({
    mutationFn: ({ id, values }: { id: string; values: Partial<Chat> }) => api.updateChat(id, values),
    onMutate: ({ id, values }) => {
      void client.cancelQueries({ queryKey: ["chat", id] });
      const previousChat = client.getQueryData<ChatDetail>(["chat", id]);
      const previousChats = client.getQueryData<Chat[]>(["chats"]);
      client.setQueryData<ChatDetail>(["chat", id], (current) => (
        current ? { ...current, ...values } : current
      ));
      client.setQueryData<Chat[]>(["chats"], (current) => current?.map((item) => (
        item.id === id ? { ...item, ...values } : item
      )));
      return { previousChat, previousChats };
    },
    onError: (_error, { id }, context) => {
      setChatDrafts((current) => {
        const next = { ...current };
        delete next[id];
        return next;
      });
      if (context?.previousChat) client.setQueryData(["chat", id], context.previousChat);
      if (context?.previousChats) client.setQueryData(["chats"], context.previousChats);
    },
    onSuccess: (updated, { id, values }) => {
      if (updated) {
        client.setQueryData<ChatDetail>(["chat", id], (current) => (
          current ? { ...current, ...updated } : current
        ));
        client.setQueryData<Chat[]>(["chats"], (current) => current?.map((item) => (
          item.id === id ? { ...item, ...updated } : item
        )));
        setChatDrafts((current) => {
          const draft = current[id];
          if (!draft) return current;
          const remaining = { ...draft };
          for (const key of Object.keys(values) as (keyof Chat)[]) {
            if (remaining[key] === values[key]) delete remaining[key];
          }
          const next = { ...current };
          if (Object.keys(remaining).length) next[id] = remaining;
          else delete next[id];
          return next;
        });
      }
    },
    onSettled: (updated, error, { id }) => {
      if (updated || error) {
        void client.invalidateQueries({ queryKey: ["chat", id] });
        void client.invalidateQueries({ queryKey: ["chats"] });
      }
    },
  });
  const manageChat = useMutation({
    mutationFn: ({ id, values }: { id: string; values: Partial<Chat> }) => api.updateChat(id, values),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["chat"] });
      void client.invalidateQueries({ queryKey: ["chats"] });
    },
  });
  const deleteChat = useMutation({
    mutationFn: ({ id, deleteGeneratedMedia }: { id: string; deleteGeneratedMedia: boolean }) => api.deleteChat(id, deleteGeneratedMedia),
    onMutate: async ({ id: deletedId }) => {
      await client.cancelQueries({ queryKey: ["chats"] });
      const previousChats = client.getQueryData<Chat[]>(["chats"]) ?? [];
      const remainingChats = previousChats.filter((candidate) => candidate.id !== deletedId);
      const previousCurrentChatId = currentChatId;
      client.setQueryData<Chat[]>(["chats"], remainingChats);
      if (activeChatId === deletedId) {
        const nextChatId = remainingChats.find((candidate) => !candidate.archived)?.id ?? null;
        setCurrentChatId(nextChatId);
        if (nextChatId) localStorage.setItem(CURRENT_CHAT_KEY, nextChatId);
        else localStorage.removeItem(CURRENT_CHAT_KEY);
      }
      client.removeQueries({ queryKey: ["chat", deletedId], exact: true });
      return { previousChats, previousCurrentChatId };
    },
    onSuccess: (_value, { id: deletedId }) => {
      setChatDrafts((current) => {
        const next = { ...current };
        delete next[deletedId];
        return next;
      });
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (_error, _deletedChat, context) => {
      if (!context) return;
      client.setQueryData(["chats"], context.previousChats);
      setCurrentChatId(context.previousCurrentChatId);
      if (context.previousCurrentChatId) localStorage.setItem(CURRENT_CHAT_KEY, context.previousCurrentChatId);
      else localStorage.removeItem(CURRENT_CHAT_KEY);
    },
    onSettled: () => void client.invalidateQueries({ queryKey: ["chats"] }),
  });
  const { updateProject, deleteProject, exportProject, importProject } = useProjectMutations({
    client,
    onImportedChat: (chatId) => {
      setCurrentChatId(chatId);
      localStorage.setItem(CURRENT_CHAT_KEY, chatId);
      setView("chat");
    },
  });

  const openLibraryEdit = useCallback((artifacts: ArtifactLibraryItem[]) => {
    const target = libraryEditTarget(artifacts);
    if (!target) return;
    setLibraryEdit(target);
    if (!activeChatId) createChat.mutate(null);
    setView("chat");
    focusMainContent();
  }, [activeChatId, createChat]);

  const allChats = useMemo(() => chats.data ?? [], [chats.data]);
  const allProjects = useMemo(() => projects.data ?? [], [projects.data]);
  const activeContent = useMemo(() => {
    if (view === "media") return <MediaLibraryView onEditImages={openLibraryEdit} />;
    if (view === "models") return <ModelsView key={modelLibraryRole} initialRole={modelLibraryRole} />;
    if (view === "workflows") return <WorkflowsView />;
    if (view === "settings") return <SettingsView engines={engines.data ?? []} />;
    const displayedChat = chat.data
      ? { ...chat.data, ...(chatDrafts[chat.data.id] ?? {}) }
      : undefined;
    const selectedRole = roleForMode(displayedChat?.routing_mode ?? "auto");
    const scopedSettings = displayedChat?.generation_settings_json?.[selectedRole] ?? {};
    const presetId = displayedChat?.generation_preset_ids_json?.[selectedRole] ?? null;
    const persistActiveChat = (values: Partial<Chat>) => {
      if (!displayedChat) return;
      setChatDrafts((current) => ({
        ...current,
        [displayedChat.id]: { ...(current[displayedChat.id] ?? {}), ...values },
      }));
      client.setQueryData<ChatDetail>(["chat", displayedChat.id], (current) => (
        current ? { ...current, ...values } : current
      ));
      updateChat.mutate({ id: displayedChat.id, values });
    };
    return <ChatView key={displayedChat?.id ?? "empty-chat"} libraryEdit={libraryEdit} chat={displayedChat} engines={engines.data ?? []} profiles={profiles.data ?? []} presets={presets.data ?? []} workflows={workflows.data ?? []} project={allProjects.find((item) => item.id === displayedChat?.project_id)} liveText={liveText} pendingTurns={displayedChat ? pendingTurns[displayedChat.id] ?? [] : []} workPlans={workPlans.data ?? []} settings={scopedSettings} presetId={presetId} onSettings={(settings) => {
      if (!displayedChat) return;
      const role = roleForMode(displayedChat.routing_mode);
      persistActiveChat({
        generation_settings_json: {
          ...(displayedChat.generation_settings_json ?? {}),
          [role]: settings,
        },
      });
    }} onPreset={(selectedPresetId) => {
      if (!displayedChat) return;
      const role = roleForMode(displayedChat.routing_mode);
      const bindings = { ...(displayedChat.generation_preset_ids_json ?? {}) };
      if (selectedPresetId) bindings[role] = selectedPresetId;
      else delete bindings[role];
      persistActiveChat({ generation_preset_ids_json: bindings });
    }} onMode={(mode) => {
      persistActiveChat({ routing_mode: mode });
    }} onProfile={(field, id) => {
      persistActiveChat({ [field]: id });
    }} onRegenerate={(messageId, settings) => {
      if (displayedChat) regenerate.mutate({ chatId: displayedChat.id, messageId, settings });
    }} onSelectRevision={(messageId, revisionId) => {
      if (displayedChat) {
        selectResponseRevision.mutate({
          chatId: displayedChat.id,
          messageId,
          revisionId,
        });
      }
    }} onEdit={(messageId, text, mode, settings) => {
      if (displayedChat) branch.mutate({
        chatId: displayedChat.id,
        messageId,
        text,
        mode,
        settings,
      });
    }} onStop={() => {
      if (displayedChat) stop.mutate(displayedChat.id);
    }} onStopAndSend={(text, mode, artifacts, settings) => {
      if (displayedChat) {
        send.mutate({
          chatId: displayedChat.id,
          id: crypto.randomUUID(),
          text,
          mode,
          artifacts,
          settings,
          stopCurrent: true,
        });
      }
    }} onDeleteExchange={deleteExchange.mutate} onForkThread={forkThread.mutate} onCancelPlan={(planId) => {
      cancelWorkPlan.mutate(planId);
    }} onCancelStep={(stepId) => {
      cancelWorkStep.mutate(stepId);
    }} onRetryStep={(stepId) => {
      retryWorkStep.mutate(stepId);
    }} onSend={(text, mode, artifacts, settings) => {
      if (displayedChat) {
        send.mutate({
          chatId: displayedChat.id,
          id: crypto.randomUUID(),
          text,
          mode,
          artifacts,
          settings,
        });
      }
    }} />;
  }, [view, modelLibraryRole, engines.data, profiles.data, presets.data, workflows.data, allProjects, chat.data, chatDrafts, liveText, pendingTurns, workPlans.data, send, regenerate, selectResponseRevision, branch, stop, cancelWorkPlan, cancelWorkStep, retryWorkStep, updateChat, deleteExchange, forkThread, client, libraryEdit, openLibraryEdit]);

  if (firstRunSetup && setupReadiness.data) {
    return <FirstRunSetup report={setupReadiness.data} onExit={exitFirstRunSetup} onOpenModels={(role) => { exitFirstRunSetup(); setModelLibraryRole(role); setView("models"); }} onOpenWorkflows={() => { exitFirstRunSetup(); setView("workflows"); }} />;
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <Sidebar projects={allProjects} chats={allChats} engines={engines.data ?? []} presets={presets.data ?? []} workflows={workflows.data ?? []} currentChatId={activeChatId} view={view} setupState={setupReadiness.data?.state} onSetup={() => setSetupOpen(true)} onChat={(id) => { setCurrentChatId(id); localStorage.setItem(CURRENT_CHAT_KEY, id); setView("chat"); focusMainContent(); }} onView={(nextView) => { setView(nextView); focusMainContent(); }} onNewChat={(projectId) => createChat.mutate(projectId)} onNewProject={() => { const name = window.prompt("Project name"); if (name?.trim()) createProject.mutate(name.trim()); }} onExportProject={(id, includeMedia) => exportProject.mutate({ id, includeMedia })} onImportProject={(file) => importProject.mutate(file)} onUpdateChat={(id, values) => manageChat.mutate({ id, values })} onDeleteChat={(id, deleteGeneratedMedia) => deleteChat.mutate({ id, deleteGeneratedMedia })} onUpdateProject={(id, values) => updateProject.mutate({ id, values })} onDeleteProject={(id) => deleteProject.mutate(id)} />
      <main id="main-content" tabIndex={-1}>{activeContent}</main>
      {setupOpen === true && !setupReadiness.data && (
        <AccessibleDialog
          title="Checking local setup"
          eyebrow="Local models"
          closeLabel="Close setup"
          onClose={() => {
            sessionStorage.setItem(SETUP_DISMISSED_KEY, "1");
            setSetupOpen(false);
          }}
          className="setup-wizard"
        >
          {setupReadiness.error
            ? <ErrorCallout message={setupReadiness.error.message} action={<button className="secondary compact-button" onClick={() => void setupReadiness.refetch()}>Retry</button>} />
            : <div className="submission-progress"><LoaderCircle size={17} /><span>Checking models and runtimes…</span></div>}
          <footer><button className="secondary" onClick={() => setSetupOpen(false)}>Not now</button></footer>
        </AccessibleDialog>
      )}
      {setupVisible && setupReadiness.data && (
        <SetupWizard
          report={setupReadiness.data}
          onClose={() => {
            sessionStorage.setItem(SETUP_DISMISSED_KEY, "1");
            setSetupOpen(false);
          }}
          onOpenModels={(role) => {
            setModelLibraryRole(role);
            setView("models");
            setSetupOpen(false);
            focusMainContent();
          }}
          onOpenWorkflows={() => {
            setView("workflows");
            setSetupOpen(false);
            focusMainContent();
          }}
        />
      )}
      <JobsPanel />
      <GlobalNotices connected={eventsConnected} mutations={[send, regenerate, selectResponseRevision, branch, stop, cancelWorkPlan, cancelWorkStep, retryWorkStep, updateChat, createChat, createProject, exportProject, importProject, manageChat, deleteChat, updateProject, deleteProject]} />
    </div>
  );
}
