import { useEffect, useMemo, useRef, useState } from "react";
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  Bot,
  ChevronDown,
  CircleStop,
  Cpu,
  Download,
  Film,
  Folder,
  Gauge,
  HardDrive,
  Image as ImageIcon,
  Library,
  Menu,
  MessageSquare,
  MoreHorizontal,
  Paperclip,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Search,
  Send,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Workflow as WorkflowIcon,
  X,
} from "lucide-react";
import { api, connectEvents } from "./api";
import {
  normalizeSettingsForFields,
  resolveCapabilitySettings,
  resolveWorkflowSettings,
} from "./settings";
import type {
  AppEvent,
  CatalogModel,
  Chat,
  ChatDetail,
  EngineCapabilities,
  GenerationPreset,
  GenerationPresetBundle,
  Message,
  MessagePart,
  ModelProfile,
  ModelProfileBundle,
  Project,
  ReferenceRecipe,
  RoutingMode,
  SettingField,
  Workflow,
  WorkflowBundle,
} from "./types";

type View = "chat" | "media" | "models" | "workflows" | "settings";
type Visibility = "basic" | "advanced" | "expert";

const visibilityRank: Record<Visibility, number> = { basic: 0, advanced: 1, expert: 2 };
const AUTO_PROFILE_ID = "__auto__";

function formatBytes(value?: number | null): string {
  if (value == null) return "Unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size >= 10 || index === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
}

function downloadJson(value: unknown, filename: string): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function EmptyState({ icon, title, body }: { icon: React.ReactNode; title: string; body: string }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">{icon}</div>
      <h2>{title}</h2>
      <p>{body}</p>
    </div>
  );
}

function AtelierMark() {
  return (
    <svg viewBox="0 0 400 400" role="img" aria-label="LM Atelier">
      <path className="lma-l" d="M43 20h64v300h169v60H43z" />
      <path className="lma-ma" d="M125 20l118 109L356 20v360h-58v-92h-85l46-45h39v-82L164 303h-39z" />
    </svg>
  );
}

function StatusDot({ healthy }: { healthy: boolean }) {
  return <span className={`status-dot ${healthy ? "healthy" : "offline"}`} />;
}

function ArtifactPart({ part }: { part: MessagePart }) {
  if (!part.artifact_id) return null;
  const proxyId = typeof part.metadata_json.browser_proxy_artifact_id === "string" ? part.metadata_json.browser_proxy_artifact_id : null;
  const source = `/api/artifacts/${encodeURIComponent(proxyId ?? part.artifact_id)}/content`;
  const preview = Boolean(part.metadata_json.preview);
  if (part.type === "image") {
    return (
      <figure className={`media-card ${preview ? "preview" : ""}`}>
        <img src={source} alt={preview ? "Generation preview" : "Generated result"} loading="lazy" />
        <figcaption>
          <ImageIcon size={14} /> {preview ? "Generation preview" : "Generated image"}
        </figcaption>
      </figure>
    );
  }
  const posterId = typeof part.metadata_json.poster_artifact_id === "string"
    ? part.metadata_json.poster_artifact_id
    : null;
  const poster = posterId ? `/api/artifacts/${encodeURIComponent(posterId)}/content` : undefined;
  return (
    <figure className="media-card">
      <video src={source} poster={poster} controls preload="metadata" />
      <figcaption>
        <Film size={14} /> Generated video
      </figcaption>
    </figure>
  );
}

function MediaLibraryView() {
  const client = useQueryClient();
  const [kind, setKind] = useState("");
  const [search, setSearch] = useState("");
  const artifacts = useQuery({
    queryKey: ["artifacts", kind, search],
    queryFn: () => api.artifacts(kind, search),
  });
  const storage = useQuery({ queryKey: ["artifact-storage"], queryFn: api.artifactStorage });
  const cleanup = useMutation({
    mutationFn: () => api.cleanupArtifacts(false),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
    },
  });
  return (
    <div className="page-view media-library">
      <header className="page-header">
        <div><small>Local artifacts</small><h1>Media library</h1><p>Browse generated images and videos across every chat and project.</p></div>
      </header>
      {storage.data && <section className={`artifact-storage-summary ${storage.data.warning ? "warning" : ""}`}>
        <div><strong>{formatBytes(storage.data.total_bytes)}</strong><small>{storage.data.total_count} stored artifacts</small></div>
        <div><strong>{formatBytes(storage.data.referenced_bytes)}</strong><small>{storage.data.referenced_count} referenced</small></div>
        <div><strong>{formatBytes(storage.data.disk_free_bytes)}</strong><small>disk available</small></div>
        <div><strong>{formatBytes(storage.data.eligible_bytes)}</strong><small>{storage.data.eligible_count} eligible for cleanup</small></div>
        <button className="secondary" disabled={!storage.data.eligible_count || cleanup.isPending} onClick={() => cleanup.mutate()}>{cleanup.isPending ? "Cleaning…" : "Clean eligible"}</button>
      </section>}
      <div className="media-toolbar">
        <div className="workspace-search"><Search size={14} /><input aria-label="Search media" placeholder="Search filenames or hashes" value={search} onChange={(event) => setSearch(event.target.value)} /></div>
        <select aria-label="Media type" value={kind} onChange={(event) => setKind(event.target.value)}><option value="">Images and videos</option><option value="image">Images</option><option value="video">Videos</option></select>
      </div>
      {cleanup.data && <div className="callout success">Removed {cleanup.data.removed_count} artifact{cleanup.data.removed_count === 1 ? "" : "s"} and reclaimed {formatBytes(cleanup.data.reclaimed_bytes)}. Newly unreferenced artifacts begin a {storage.data?.retention_days ?? 30}-day recovery window.</div>}
      {cleanup.error && <div className="callout error">{cleanup.error.message}</div>}
      {artifacts.data?.length ? <div className="media-grid">{artifacts.data.map((artifact) => {
        const source = `/api/artifacts/${encodeURIComponent(artifact.id)}/content`;
        const proxyId = typeof artifact.metadata_json.browser_proxy_artifact_id === "string" ? artifact.metadata_json.browser_proxy_artifact_id : null;
        const playbackSource = proxyId ? `/api/artifacts/${encodeURIComponent(proxyId)}/content` : source;
        const posterId = typeof artifact.metadata_json.poster_artifact_id === "string" ? artifact.metadata_json.poster_artifact_id : null;
        return <article className="gallery-card" key={artifact.id}>
          {artifact.kind === "image" ? <img src={source} alt={artifact.original_name ?? "Generated image"} loading="lazy" /> : <video src={playbackSource} poster={posterId ? `/api/artifacts/${encodeURIComponent(posterId)}/content` : undefined} controls preload="metadata" />}
          <div><strong>{artifact.original_name ?? artifact.kind}</strong><small>{formatBytes(artifact.size_bytes)} · {artifact.reference_count} reference{artifact.reference_count === 1 ? "" : "s"}</small><span><a href={source} download>Download</a><code>{artifact.sha256.slice(0, 12)}</code></span></div>
        </article>;
      })}</div> : <EmptyState icon={<ImageIcon />} title="No generated media" body="Images and videos created in chat will appear here." />}
    </div>
  );
}

function PartView({ part, liveText, markdown = false }: { part: MessagePart; liveText?: string; markdown?: boolean }) {
  if (part.type === "text") {
    const text = liveText || part.text || "";
    return markdown ? <div className="message-text markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown></div> : <div className="message-text">{text}</div>;
  }
  if (part.type === "image" || part.type === "video") return <ArtifactPart part={part} />;
  if (part.type === "progress") {
    const progress = Number(part.metadata_json.progress ?? 0);
    return (
      <div className="generation-progress">
        <Sparkles size={17} />
        <div>
          <span>{part.text || "Working"}</span>
          <div className="progress-track"><div style={{ width: `${Math.max(4, progress * 100)}%` }} /></div>
        </div>
      </div>
    );
  }
  if (part.type === "error") return <div className="message-error">{part.text}</div>;
  return <div className="message-error">Unsupported message part: {String(part.type)}</div>;
}

function MessageBubble({
  message,
  liveText,
  onRegenerate,
  onEdit,
}: {
  message: Message;
  liveText?: string;
  onRegenerate?: (messageId: string) => void;
  onEdit?: (messageId: string, text: string) => void;
}) {
  const visibleParts = message.parts.filter((part) => part.type !== "generation_metadata");
  const userText = visibleParts.filter((part) => part.type === "text").map((part) => part.text || "").join("\n");
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(userText);
  const metadata = message.parts.find((part) => part.type === "generation_metadata")?.metadata_json;
  const context = metadata?.context as Record<string, unknown> | undefined;
  const provenance = metadata?.provenance as Record<string, unknown> | undefined;
  const modelSelection = provenance?.model_selection as Record<string, unknown> | undefined;
  const autoProfileName = modelSelection?.mode === "auto"
    ? String(modelSelection.profile_name ?? "")
    : "";
  const usage = context?.usage as Record<string, unknown> | undefined;
  const inputTokens = Number(usage?.prompt_tokens ?? context?.input_tokens ?? 0);
  const contextLimit = Number(context?.context_limit ?? 0);
  const omitted = Number(context?.messages_omitted ?? 0);
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "You" : <Bot size={19} />}</div>
      <div className="message-content">
        {editing ? <div className="message-edit"><textarea aria-label="Edit message" rows={4} value={draft} onChange={(event) => setDraft(event.target.value)} /><div><button onClick={() => { setDraft(userText); setEditing(false); }}>Cancel</button><button className="primary" disabled={!draft.trim()} onClick={() => { onEdit?.(message.id, draft.trim()); setEditing(false); }}>Send edited message</button></div></div> : visibleParts.map((part) => <PartView key={part.id} part={part} liveText={liveText} markdown={message.role === "assistant"} />)}
        {liveText && !visibleParts.some((part) => part.type === "text") && (
          <div className="message-text markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{liveText}</ReactMarkdown></div>
        )}
        {message.role === "user" && message.status === "complete" && onEdit && !editing && <div className="message-meta"><button onClick={() => setEditing(true)}>Edit and branch</button></div>}
        {message.role === "assistant" && message.status === "cancelled" && !visibleParts.some((part) => part.type === "error") && (
          <div className="message-meta"><span>Generation cancelled</span></div>
        )}
        {message.role === "assistant" && message.status === "complete" && (
          <div className="message-meta">
            {autoProfileName && <span>Auto chose {autoProfileName}</span>}
            {contextLimit > 0 && (
              <span>
                Context {inputTokens.toLocaleString()} / {contextLimit.toLocaleString()} tokens
                {omitted > 0 ? ` · ${omitted} earlier message${omitted === 1 ? "" : "s"} omitted` : ""}
              </span>
            )}
            {onRegenerate && (
              <button onClick={() => onRegenerate(message.id)} aria-label="Regenerate response">
                <RotateCcw size={13} /> Regenerate
              </button>
            )}
          </div>
        )}
      </div>
    </article>
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
  if (field.type === "boolean") {
    return (
      <label className="setting-row toggle-row">
        <span><strong>{field.label}</strong><small>{field.help}</small></span>
        <input type="checkbox" checked={Boolean(value)} disabled={fixed} onChange={(event) => onChange(event.target.checked)} />
      </label>
    );
  }
  if (field.type === "enum") {
    return (
      <label className="setting-row">
        <span><strong>{field.label}</strong><small>{field.help}</small></span>
        <select value={String(value ?? "")} disabled={fixed} onChange={(event) => onChange(event.target.value)}>
          {field.choices.map((choice) => <option key={String(choice)}>{String(choice)}</option>)}
        </select>
      </label>
    );
  }
  if (field.type === "number" || field.type === "integer") {
    return (
      <label className="setting-row">
        <span><strong>{field.label}</strong><small>{field.help}</small></span>
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
      <label className="setting-row">
        <span><strong>{field.label}</strong><small>{field.help}</small></span>
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
    <label className="setting-row">
      <span><strong>{field.label}</strong><small>{field.help}</small></span>
      <input value={String(value ?? "")} disabled={fixed} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function SettingsDrawer({
  open,
  onClose,
  mode,
  engines,
  values,
  onValues,
  workflowSchema,
}: {
  open: boolean;
  onClose: () => void;
  mode: RoutingMode;
  engines: EngineCapabilities[];
  values: Record<string, unknown>;
  onValues: (values: Record<string, unknown>) => void;
  workflowSchema?: Record<string, unknown>;
}) {
  const [visibility, setVisibility] = useState<Visibility>("basic");
  const role = mode === "video" ? "video" : mode === "image" ? "image" : "chat";
  const engine = engines.find((item) => item.roles.includes(role));
  const fields = resolveWorkflowSettings(
    resolveCapabilitySettings(engine, role),
    workflowSchema,
  ).filter(
    (field) =>
      field.scope !== "load"
      && visibilityRank[field.visibility] <= visibilityRank[visibility]
      && field.available,
  );
  return (
    <aside className={`settings-drawer ${open ? "open" : ""}`} aria-hidden={!open}>
      <header>
        <div><small>Turn controls</small><h2>{role[0].toUpperCase() + role.slice(1)} settings</h2></div>
        <button className="icon-button" onClick={onClose} aria-label="Close settings"><X /></button>
      </header>
      <div className="segmented compact">
        {(["basic", "advanced", "expert"] as Visibility[]).map((level) => (
          <button key={level} className={visibility === level ? "active" : ""} onClick={() => setVisibility(level)}>{level}</button>
        ))}
      </div>
      <div className="settings-list">
        {fields.map((field) => (
          <SettingControl
            key={`${field.scope}:${field.key}`}
            field={field}
            value={values[field.key] ?? field.default}
            onChange={(value) => onValues({ ...values, [field.key]: value })}
          />
        ))}
        {!engine && <p className="muted">No {role} engine is configured.</p>}
      </div>
      <footer><button className="secondary" onClick={() => onValues({})}>Reset turn overrides</button></footer>
    </aside>
  );
}

function Composer({
  chat,
  engines,
  busy,
  settings,
  onSettings,
  onSend,
  onStop,
  workflows,
  project,
}: {
  chat: Chat;
  engines: EngineCapabilities[];
  busy: boolean;
  settings: Record<string, unknown>;
  onSettings: (settings: Record<string, unknown>) => void;
  onSend: (text: string, mode: RoutingMode, artifacts: string[], settings: Record<string, unknown>) => void;
  onStop: () => void;
  workflows: Workflow[];
  project?: Project;
}) {
  const [text, setText] = useState("");
  const [mode, setMode] = useState<RoutingMode>(chat.routing_mode);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [attachments, setAttachments] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const workflowSchema = workflowSchemaForTurn(workflows, project, mode, attachments.length > 0);

  const submit = () => {
    if (!text.trim() || busy) return;
    const role = mode === "video" ? "video" : mode === "image" ? "image" : "chat";
    const engine = engines.find((item) => item.roles.includes(role));
    const fields = resolveWorkflowSettings(
      resolveCapabilitySettings(engine, role),
      workflowSchema,
    );
    onSend(text.trim(), mode, attachments, normalizeSettingsForFields(settings, fields));
    setText("");
    setAttachments([]);
  };

  const upload = async (file?: File) => {
    if (!file) return;
    setUploading(true);
    try {
      const id = await api.upload(file);
      setAttachments((current) => [...current, id]);
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      <div className="composer-wrap">
        {attachments.length > 0 && (
          <div className="attachment-strip">
            {attachments.map((id) => <span key={id}><Paperclip size={13} />{id.slice(0, 18)}<button onClick={() => setAttachments((items) => items.filter((item) => item !== id))}><X size={12} /></button></span>)}
          </div>
        )}
        <div className="composer">
          <textarea
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
              <input ref={fileInput} hidden type="file" accept="image/*,video/*" onChange={(event) => void upload(event.target.files?.[0])} />
              <label className={`mode-select mode-${mode}`}>
                {mode === "auto" && <Sparkles size={15} />}
                {mode === "text" && <MessageSquare size={15} />}
                {mode === "image" && <ImageIcon size={15} />}
                {mode === "video" && <Film size={15} />}
                <select value={mode} onChange={(event) => setMode(event.target.value as RoutingMode)}>
                  <option value="auto">Auto</option><option value="text">Text</option><option value="image">Image</option><option value="video">Video</option>
                </select>
                <ChevronDown size={13} />
              </label>
              <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Turn settings"><SlidersHorizontal size={18} /></button>
            </div>
            {busy ? <button className="send-button stop" onClick={onStop} aria-label="Stop response"><CircleStop size={18} /></button> : <button className="send-button" disabled={!text.trim()} onClick={submit} aria-label="Send"><Send size={18} /></button>}
          </div>
        </div>
        <small className="composer-note">Local models can make mistakes. Generation stays on this machine.</small>
      </div>
      <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} mode={mode} engines={engines} values={settings} onValues={onSettings} workflowSchema={workflowSchema} />
    </>
  );
}

function activeBranchMessages(chat: ChatDetail): Message[] {
  if (!chat.active_head_message_id) return chat.messages;
  const byId = new Map(chat.messages.map((message) => [message.id, message]));
  const lineage: Message[] = [];
  const visited = new Set<string>();
  let current = byId.get(chat.active_head_message_id);
  while (current && !visited.has(current.id)) {
    visited.add(current.id);
    lineage.unshift(current);
    current = current.parent_id ? byId.get(current.parent_id) : undefined;
  }
  return lineage.length > 0 ? lineage : chat.messages;
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
  settings,
  onSettings,
  onSend,
  onProfile,
  onRegenerate,
  onEdit,
  onStop,
}: {
  chat?: ChatDetail;
  engines: EngineCapabilities[];
  profiles: ModelProfile[];
  workflows: Workflow[];
  project?: Project;
  liveText: Record<string, string>;
  settings: Record<string, unknown>;
  onSettings: (settings: Record<string, unknown>) => void;
  onSend: (text: string, mode: RoutingMode, artifacts: string[], settings: Record<string, unknown>) => void;
  onProfile: (field: "active_chat_profile_id" | "active_image_profile_id" | "active_video_profile_id", id: string | null) => void;
  onRegenerate: (messageId: string, settings: Record<string, unknown>) => void;
  onEdit: (messageId: string, text: string, settings: Record<string, unknown>) => void;
  onStop: () => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (typeof endRef.current?.scrollIntoView === "function") {
      endRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chat?.messages, liveText]);
  if (!chat) return <EmptyState icon={<MessageSquare />} title="Start a local conversation" body="Create a chat, choose your models, and keep every response on your machine." />;
  const messages = activeBranchMessages(chat);
  const busy = messages.some((message) => message.status === "pending");
  return (
    <div className="chat-view">
      <div className="chat-header">
        <div><small>{chat.project_id ? "Project chat" : "Unfiled chat"}</small><h1>{chat.title}</h1></div>
        <div className="chat-profile-selectors">
          {(["chat", "image", "video"] as const).map((role) => {
            const field = `active_${role}_profile_id` as "active_chat_profile_id" | "active_image_profile_id" | "active_video_profile_id";
            const selected = profiles.find((profile) => profile.id === chat[field]);
            const value = selected?.is_default ? "" : chat[field] ?? "";
            return <label key={role}><span>{role}</span><select value={value} title={`Choose the ${role} model for this chat`} onChange={(event) => onProfile(field, event.target.value || null)}><option value={AUTO_PROFILE_ID}>Auto</option><option value="">Default</option>{profiles.filter((profile) => profile.role === role && !profile.is_default).map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}</select></label>;
          })}
          <button className="icon-button"><MoreHorizontal /></button>
        </div>
      </div>
      <div className="messages">
        {messages.length === 0 ? (
          <EmptyState icon={<Sparkles />} title="What should we make?" body="Ask a question or describe an image or video. Auto mode chooses the appropriate local model." />
        ) : messages.map((message) => <MessageBubble
          key={message.id}
          message={message}
          liveText={liveText[message.id]}
          onRegenerate={busy ? undefined : (messageId) => onRegenerate(messageId, settings)}
          onEdit={busy ? undefined : (messageId, text) => onEdit(messageId, text, settings)}
        />)}
        <div ref={endRef} />
      </div>
      <Composer chat={chat} engines={engines} busy={busy} settings={settings} onSettings={onSettings} onSend={onSend} onStop={onStop} workflows={workflows} project={project} />
    </div>
  );
}

function ModelCard({
  model,
  role,
  onDownload,
  status,
}: {
  model: CatalogModel;
  role: string;
  onDownload: () => void;
  status: "idle" | "preparing" | "downloading" | "installed";
}) {
  const label = {
    idle: "Install",
    preparing: "Preparing…",
    downloading: "Downloading…",
    installed: "Installed",
  }[status];
  return (
    <article className="model-card">
      <div className="model-icon">{role === "video" ? <Film /> : role === "image" ? <ImageIcon /> : <Bot />}</div>
      <div className="model-copy">
        <h3>{model.name}</h3><p>{model.author} · {model.pipeline_tag || model.library_name || "model"}</p>
        <div className="badges"><span className={`badge ${model.compatibility}`}>{model.compatibility.replace("_", " ")}</span>{model.gated && <span className="badge">Gated</span>}{model.formats.slice(0, 2).map((format) => <span className="badge" key={format}>{format}</span>)}{model.quantizations.slice(0, 2).map((value) => <span className="badge" key={value}>{value}</span>)}</div>
        <small>{model.compatibility_reasons.join(" · ")}</small>
      </div>
      <div className="model-stats"><span><Download size={14} />{model.downloads?.toLocaleString() ?? "—"}</span><button className="primary compact-button" onClick={onDownload} disabled={status !== "idle" || model.compatibility === "unsupported"}>{label}</button></div>
    </article>
  );
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
      <div className="recipe-badges"><span className="badge likely">Reference candidate</span><span className="badge">{recipe.license_id}</span><span className="badge">{recipe.node_policy || recipe.engine}</span></div>
      <div className="recipe-meta"><span><HardDrive size={14} />{formatBytes(recipe.total_size_bytes)}</span><span><Gauge size={14} />{memory}</span></div>
      <small>{recipe.hardware.guidance}</small>
      <button className="primary" onClick={onInstall} disabled={pending}>{pending ? "Queued" : "Install pinned recipe"}</button>
    </article>
  );
}

function ModelsView() {
  const client = useQueryClient();
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [role, setRole] = useState("chat");
  const [sort, setSort] = useState("trending");
  const [compatibility, setCompatibility] = useState("");
  const [fileFormat, setFileFormat] = useState("");
  const [gated, setGated] = useState("");
  const [quantization, setQuantization] = useState("");
  const [licenseId, setLicenseId] = useState("");
  const [architecture, setArchitecture] = useState("");
  const [minParameters, setMinParameters] = useState("");
  const [maxParameters, setMaxParameters] = useState("");
  const [maxSizeGb, setMaxSizeGb] = useState("");
  const [importOpen, setImportOpen] = useState(false);
  const [importName, setImportName] = useState("");
  const [importPath, setImportPath] = useState("");
  const [importRole, setImportRole] = useState("chat");
  const [importEngine, setImportEngine] = useState("llama.cpp");
  const catalogFilters = {
    compatibility,
    file_format: fileFormat,
    gated,
    quantization,
    license_id: licenseId,
    architecture,
    min_parameters: minParameters ? String(Number(minParameters) * 1_000_000_000) : "",
    max_parameters: maxParameters ? String(Number(maxParameters) * 1_000_000_000) : "",
    max_size_bytes: maxSizeGb ? String(Number(maxSizeGb) * 1024 ** 3) : "",
  };
  const catalog = useInfiniteQuery({
    queryKey: ["catalog", submitted, role, sort, catalogFilters],
    queryFn: ({ pageParam }) => api.catalog(submitted, role, sort, pageParam, catalogFilters),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });
  const catalogItems = useMemo(() => catalog.data?.pages.flatMap((page) => page.items) ?? [], [catalog.data]);
  const recipes = useQuery({ queryKey: ["recipes"], queryFn: api.recipes });
  const installed = useQuery({ queryKey: ["models"], queryFn: api.models });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3_000 });
  const storage = useQuery({ queryKey: ["model-storage"], queryFn: api.modelStorage });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const download = useMutation({
    mutationFn: async ({ model, selectedRole }: { model: CatalogModel; selectedRole: string }) => {
      const engine = selectedRole === "chat" ? "llama.cpp" : "comfyui";
      const preflight = await api.catalogPreflight(
        model.remote_id,
        selectedRole,
        engine,
        "main",
        [],
      );
      if (!preflight.can_install) {
        const blockers = preflight.checks
          .filter((check) => check.status === "block")
          .map((check) => check.detail);
        throw new Error(blockers.join(" ") || "This model cannot be installed safely.");
      }
      return api.download(
        preflight.remote_id,
        preflight.source_remote_id,
        selectedRole,
        engine,
        preflight.revision,
        preflight.selected_files,
        preflight.expected_sha256,
        preflight.comfy_paths,
        preflight.workflow_template_id,
        preflight.workflow_template_sha256,
      );
    },
    onSuccess: () => void client.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const installRecipe = useMutation({
    mutationFn: (recipeId: string) => api.installRecipe(recipeId),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const createProfile = useMutation({
    mutationFn: api.createProfile,
    onSuccess: () => void client.invalidateQueries({ queryKey: ["profiles"] }),
  });
  const deleteModel = useMutation({
    mutationFn: (modelId: string) => api.deleteModel(modelId, true),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["profiles"] });
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
    installed.data
      ?.filter((model) => model.role === role)
      ?.flatMap((model) => [
        model.manifest_json.remote_id,
        model.manifest_json.source_remote_id,
      ])
      .filter((remoteId): remoteId is string => typeof remoteId === "string") ?? [],
  );
  const activeDownloadIds = new Set(
    jobs.data
      ?.filter((job) =>
        job.kind === "download"
        && job.payload_json.role === role
        && ["queued", "running", "paused"].includes(job.status)
      )
      .map((job) => job.payload_json.remote_id)
      .filter((remoteId): remoteId is string => typeof remoteId === "string") ?? [],
  );
  return (
    <div className="page-view">
      <header className="page-header"><div><small>Model library</small><h1>Find the right local model</h1><p>Search Hugging Face, inspect compatibility, and manage downloads without leaving the app.</p></div><div className="storage-actions"><div className="storage-pill"><HardDrive size={17} />{storage.data?.installed_count ?? installed.data?.length ?? 0} installed · {formatBytes(storage.data?.installed_bytes)}</div><button className="secondary compact-button" onClick={() => setImportOpen(true)}><Folder size={16} />Import local</button><button className="secondary compact-button" disabled={!storage.data?.partial_download_count || cleanupDownloads.isPending} onClick={() => cleanupDownloads.mutate()}>Clean {storage.data?.partial_download_count ?? 0} partial</button></div></header>
      <section className="recipe-section">
        <div className="section-heading"><div><small>Curated starting points</small><h2>Reference recipes</h2></div><p>Immutable revisions, exact safe files, conservative defaults, and hardware guidance. Certification follows real-device validation.</p></div>
        {recipes.isLoading && <div className="loading-line" />}
        {recipes.error && <div className="callout error">{recipes.error.message}</div>}
        {installRecipe.error && <div className="callout error">{installRecipe.error.message}</div>}
        <div className="recipe-grid">{recipes.data?.map((recipe) => <RecipeCard key={recipe.id} recipe={recipe} pending={installRecipe.isPending && installRecipe.variables === recipe.id} onInstall={() => installRecipe.mutate(recipe.id)} />)}</div>
      </section>
      <div className="toolbar">
        <form className="search-box" onSubmit={(event) => { event.preventDefault(); setSubmitted(query); }}><Search size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models" /></form>
        <select value={role} onChange={(event) => setRole(event.target.value)}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option></select>
        <select value={sort} onChange={(event) => setSort(event.target.value)}><option value="trending">Trending</option><option value="downloads">Downloads</option><option value="likes">Likes</option><option value="newest">Newest</option><option value="updated">Recently updated</option><option value="compatible">Compatible first</option></select>
      </div>
      <div className="catalog-filters"><select aria-label="Compatibility filter" value={compatibility} onChange={(event) => setCompatibility(event.target.value)}><option value="">All compatibility</option><option value="likely">Likely compatible</option><option value="advanced_import">Advanced import</option><option value="unsupported">Unsupported</option></select><select aria-label="Format filter" value={fileFormat} onChange={(event) => setFileFormat(event.target.value)}><option value="">All formats</option><option value="gguf">GGUF</option><option value="safetensors">safetensors</option></select><select aria-label="Access filter" value={gated} onChange={(event) => setGated(event.target.value)}><option value="">All access</option><option value="open">Open</option><option value="gated">Gated</option></select><input aria-label="Quantization filter" placeholder="Quantization (Q4_K_M, FP8…)" value={quantization} onChange={(event) => setQuantization(event.target.value)} /><input aria-label="Architecture filter" placeholder="Architecture" value={architecture} onChange={(event) => setArchitecture(event.target.value)} /><input aria-label="License filter" placeholder="License" value={licenseId} onChange={(event) => setLicenseId(event.target.value)} /><input aria-label="Minimum parameters" type="number" min="0" placeholder="Min parameters (B)" value={minParameters} onChange={(event) => setMinParameters(event.target.value)} /><input aria-label="Maximum parameters" type="number" min="0" placeholder="Max parameters (B)" value={maxParameters} onChange={(event) => setMaxParameters(event.target.value)} /><input aria-label="Maximum download size" type="number" min="0" placeholder="Max download (GB)" value={maxSizeGb} onChange={(event) => setMaxSizeGb(event.target.value)} /></div>
      {(installed.data?.length ?? 0) > 0 && <section><h2>Installed models</h2><div className="profile-table model-installs">{installed.data?.map((model) => { const bound = profiles.data?.some((profile) => profile.model_install_id === model.id) ?? false; return <div key={model.id}><span className="badge">{model.role}</span><strong>{model.name}</strong><span>{formatBytes(model.size_bytes)}</span><span className="row-actions"><button className="secondary compact-button" disabled={bound || createProfile.isPending} title={bound ? "This model is available in chats and Auto mode" : "Complete setup for this older model install"} onClick={() => createProfile.mutate(model)}>{bound ? "Ready to use" : "Finish setup"}</button><button className="secondary compact-button danger" disabled={deleteModel.isPending} title="Delete installed model" onClick={() => { if (window.confirm(`Delete ${model.name} and its model settings from local storage?`)) deleteModel.mutate(model.id); }}>Delete</button></span></div>; })}</div></section>}
      {(download.error || deleteModel.error || cleanupDownloads.error) && <div className="callout error">{download.error?.message || deleteModel.error?.message || cleanupDownloads.error?.message}</div>}
      {catalog.isLoading && <div className="loading-line" />}
      {catalog.error && <div className="callout error">{catalog.error.message}</div>}
      <div className="model-grid">{catalogItems.map((model) => { const status = installedRemoteIds.has(model.remote_id) ? "installed" : activeDownloadIds.has(model.remote_id) ? "downloading" : download.isPending && download.variables?.model.remote_id === model.remote_id ? "preparing" : "idle"; return <ModelCard key={model.remote_id} model={model} role={role} status={status} onDownload={() => download.mutate({ model, selectedRole: role })} />; })}</div>
      {catalog.hasNextPage && <div className="load-more"><button className="secondary" disabled={catalog.isFetchingNextPage} onClick={() => void catalog.fetchNextPage()}>{catalog.isFetchingNextPage ? "Loading…" : "Load more models"}</button></div>}
      {importOpen && <div className="modal-backdrop"><div className="modal"><header><div><small>Advanced import</small><h2>Import a local model</h2></div><button className="icon-button" aria-label="Close local import" onClick={() => setImportOpen(false)}><X /></button></header><p>Register an existing model file or directory. Pickle-compatible formats are blocked; imported models are marked for advanced review.</p><label>Display name<input value={importName} onChange={(event) => setImportName(event.target.value)} /></label><label>Absolute local path<input value={importPath} onChange={(event) => setImportPath(event.target.value)} placeholder="/path/to/model.gguf" /></label><label>Role<select value={importRole} onChange={(event) => { const next = event.target.value; setImportRole(next); setImportEngine(next === "chat" ? "llama.cpp" : "comfyui"); }}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option></select></label><label>Runtime<select value={importEngine} onChange={(event) => setImportEngine(event.target.value)}><option value="llama.cpp">llama.cpp</option><option value="comfyui">ComfyUI</option></select></label>{importModel.error && <div className="callout error">{importModel.error.message}</div>}<footer><button className="secondary" onClick={() => setImportOpen(false)}>Cancel</button><button className="primary" disabled={!importName.trim() || !importPath.trim() || importModel.isPending} onClick={() => importModel.mutate()}>{importModel.isPending ? "Importing…" : "Import model"}</button></footer></div></div>}
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

function CustomNodesPanel() {
  const client = useQueryClient();
  const nodes = useQuery({ queryKey: ["custom-nodes"], queryFn: api.customNodes });
  const [name, setName] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [revision, setRevision] = useState("");
  const refresh = () => void client.invalidateQueries({ queryKey: ["custom-nodes"] });
  const install = useMutation({ mutationFn: () => api.installCustomNode({ name: name.trim(), source_url: sourceUrl.trim(), revision: revision.trim() }), onSuccess: () => { setName(""); setSourceUrl(""); setRevision(""); refresh(); } });
  const trust = useMutation({ mutationFn: ({ id, trusted }: { id: string; trusted: boolean }) => api.trustCustomNode(id, trusted), onSuccess: refresh });
  const update = useMutation({ mutationFn: ({ id, revision: next }: { id: string; revision: string }) => api.updateCustomNode(id, next), onSuccess: refresh });
  const rollback = useMutation({ mutationFn: api.rollbackCustomNode, onSuccess: refresh });
  const remove = useMutation({ mutationFn: api.removeCustomNode, onSuccess: refresh });
  const error = install.error || trust.error || update.error || rollback.error || remove.error;
  return <section className="custom-nodes"><div className="detail-title"><div><h2>Custom nodes</h2><p>Pinned GitHub sources stay disabled until you review their security summary and explicitly trust them. Stop ComfyUI before changing nodes.</p></div></div><div className="custom-node-install"><input aria-label="Custom node name" placeholder="Node name" value={name} onChange={(event) => setName(event.target.value)} /><input aria-label="Custom node source" placeholder="https://github.com/owner/repository" value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} /><input aria-label="Custom node commit" placeholder="Full 40-character commit SHA" value={revision} onChange={(event) => setRevision(event.target.value)} /><button className="primary" disabled={!name.trim() || !sourceUrl.trim() || revision.trim().length !== 40 || install.isPending} onClick={() => { if (window.confirm("Download this pinned repository for review? Its code will remain untrusted.")) install.mutate(); }}>Install pinned source</button></div>{error && <div className="callout error">{error.message}</div>}<div className="profile-table custom-node-list">{nodes.data?.map((node) => <div key={node.id}><span className={`badge ${node.trusted ? "likely" : "advanced_import"}`}>{node.trusted ? "Trusted" : "Review required"}</span><span><strong>{node.name}</strong><small>{node.source_url}<br />{node.revision}</small></span><details><summary>Security</summary><pre>{JSON.stringify(node.security_json, null, 2)}</pre></details><span className="row-actions"><button className="secondary compact-button" onClick={() => { const next = window.prompt("Full pinned commit SHA", node.revision); if (next?.trim() && next.trim() !== node.revision) update.mutate({ id: node.id, revision: next.trim() }); }}>Update</button>{node.previous_revision && <button className="secondary compact-button" onClick={() => rollback.mutate(node.id)}>Rollback</button>}<button className="secondary compact-button" onClick={() => node.trusted ? trust.mutate({ id: node.id, trusted: false }) : window.confirm("I reviewed this exact pinned revision and trust its code to run in ComfyUI.") && trust.mutate({ id: node.id, trusted: true })}>{node.trusted ? "Revoke trust" : "Trust revision"}</button><button className="secondary compact-button danger" onClick={() => window.confirm(`Remove ${node.name}?`) && remove.mutate(node.id)}>Remove</button></span></div>)}</div></section>;
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
  const importBundleMutation = useMutation({
    mutationFn: async (file: File) => {
      const bundle = JSON.parse(await file.text()) as WorkflowBundle;
      if (bundle.format !== "lm-atelier-workflow") throw new Error("This is not an LM Atelier workflow bundle.");
      return api.importWorkflow(bundle);
    },
    onSuccess: refresh,
  });
  const importBundle = (file?: File) => {
    if (file) importBundleMutation.mutate(file);
  };
  const openCreate = () => { setEditing(false); setName("Custom image workflow"); setDescription(""); setOperation("text_to_image"); setGraph("{}"); setUiGraph("{}"); setInputSchema("{}"); setDependencies("{}"); setTrusted(false); setNewOpen(true); };
  const openEdit = () => { if (!selected) return; const revision = selected.revisions.find((item) => item.id === selected.current_revision_id) ?? selected.revisions.at(-1); if (!revision) return; setEditing(true); setName(selected.name); setDescription(selected.description); setOperation(selected.operation); setGraph(JSON.stringify(revision.api_graph_json, null, 2)); setUiGraph(JSON.stringify(revision.ui_graph_json, null, 2)); setInputSchema(JSON.stringify(revision.input_schema_json, null, 2)); setDependencies(JSON.stringify(revision.dependencies_json, null, 2)); setTrusted(revision.trusted); setNewOpen(true); };
  const selectedRevision = selected?.revisions.find((revision) => revision.id === selectedRevisionId) ?? selected?.revisions.find((revision) => revision.id === selected.current_revision_id) ?? selected?.revisions.at(-1);
  const currentRevision = selected?.revisions.find((revision) => revision.id === selected.current_revision_id);
  return (
    <div className="page-view">
      <header className="page-header"><div><small>Workflow studio</small><h1>Media pipelines</h1><p>Version complete ComfyUI graphs and expose their inputs as reusable generation controls.</p></div><div className="storage-actions"><input ref={importInput} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0]); event.target.value = ""; }} /><button className="secondary" onClick={() => importInput.current?.click()}>Import bundle</button><button className="primary" onClick={openCreate}><Plus size={17} />New workflow</button></div></header>
      {(importBundleMutation.error || clone.error || restore.error || exportBundle.error || openInComfy.error) && <div className="callout error">{(importBundleMutation.error || clone.error || restore.error || exportBundle.error || openInComfy.error)?.message}</div>}
      {selected && <div className="storage-actions"><button className="secondary" onClick={() => openInComfy.mutate(selected.id)}>Download UI graph and open in ComfyUI</button></div>}
      <div className="workflow-layout">
        <div className="workflow-list">{workflows.data?.map((workflow) => <button key={workflow.id} className={selected?.id === workflow.id ? "selected" : ""} onClick={() => { setSelectedId(workflow.id); setSelectedRevisionId(workflow.current_revision_id); }}><WorkflowIcon size={18} /><span><strong>{workflow.name}</strong><small>{workflow.operation} · {workflow.revisions.length} revision{workflow.revisions.length === 1 ? "" : "s"}</small></span></button>)}</div>
        <div className="workflow-detail">{selected && selectedRevision ? <><div className="detail-title"><div><small>{selected.operation}</small><h2>{selected.name}</h2><p>{selected.description}</p></div><div className="row-actions"><button className="secondary compact-button" onClick={openEdit}>New revision</button><button className="secondary compact-button" onClick={() => clone.mutate(selected.id)}>Duplicate</button><button className="secondary compact-button" onClick={() => exportBundle.mutate(selected.id)}>Export</button><button className="secondary compact-button" onClick={() => validate.mutate(selected.id)}>Validate</button></div></div><div className="workflow-revision-bar"><label>Revision<select value={selectedRevision.id} onChange={(event) => setSelectedRevisionId(event.target.value)}>{[...selected.revisions].sort((a, b) => b.version - a.version).map((revision) => <option key={revision.id} value={revision.id}>v{revision.version}{revision.id === selected.current_revision_id ? " · current" : ""}</option>)}</select></label>{selectedRevision.id !== selected.current_revision_id && <button className="secondary compact-button" onClick={() => restore.mutate({ id: selected.id, revisionId: selectedRevision.id })}>Restore as new revision</button>}<span className={`badge ${selectedRevision.trusted ? "likely" : "advanced_import"}`}>{selectedRevision.trusted ? "Trusted" : "Untrusted"}</span></div><section className="workflow-input-section"><h3>Declared controls</h3><WorkflowControls schema={selectedRevision.input_schema_json} /></section><details open><summary>Executable graph</summary><pre>{JSON.stringify(selectedRevision.api_graph_json, null, 2)}</pre></details><details><summary>Dependencies</summary><pre>{JSON.stringify(selectedRevision.dependencies_json, null, 2)}</pre></details>{currentRevision && currentRevision.id !== selectedRevision.id && <details><summary>Compare with current revision</summary><div className="workflow-compare"><pre>{JSON.stringify(selectedRevision.api_graph_json, null, 2)}</pre><pre>{JSON.stringify(currentRevision.api_graph_json, null, 2)}</pre></div></details>}{validate.data && <div className={`callout ${validate.data.valid ? "success" : "error"}`}>{validate.data.valid ? "Workflow and declared dependencies are valid for the active media engine." : validate.data.errors.join("\n")}{validate.data.warnings.map((warning) => `\nWarning: ${warning}`)}</div>}</> : <EmptyState icon={<WorkflowIcon />} title="Select a workflow" body="Inspect its pinned revision, inputs, dependencies, and validation state." />}</div>
      </div>
      <CustomNodesPanel />
      {newOpen && <div className="modal-backdrop"><div className="modal workflow-editor"><header><div><small>{editing ? "Immutable revision" : "Portable workflow"}</small><h2>{editing ? "Create workflow revision" : "Create ComfyUI workflow"}</h2></div><button className="icon-button" onClick={() => setNewOpen(false)}><X /></button></header><label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>Description<textarea rows={2} value={description} onChange={(event) => setDescription(event.target.value)} /></label><label>Operation<select value={operation} disabled={editing} onChange={(event) => setOperation(event.target.value)}><option value="text_to_image">Text to image</option><option value="image_to_image">Image to image</option><option value="text_to_video">Text to video</option><option value="image_to_video">Image to video</option></select></label><label>API-format workflow JSON<textarea rows={10} value={graph} onChange={(event) => setGraph(event.target.value)} /></label><label>UI workflow JSON<textarea rows={5} value={uiGraph} onChange={(event) => setUiGraph(event.target.value)} /></label><label>Declared input schema JSON<textarea rows={6} value={inputSchema} onChange={(event) => setInputSchema(event.target.value)} /></label><label>Dependencies JSON<textarea rows={5} value={dependencies} onChange={(event) => setDependencies(event.target.value)} /></label><label className="toggle-row"><span><strong>Trust this workflow</strong><small>Only enable after reviewing every node and dependency.</small></span><input type="checkbox" checked={trusted} onChange={(event) => setTrusted(event.target.checked)} /></label>{save.error && <div className="callout error">{save.error.message}</div>}<footer><button className="secondary" onClick={() => setNewOpen(false)}>Cancel</button><button className="primary" disabled={!name.trim() || save.isPending} onClick={() => save.mutate()}>{save.isPending ? "Saving…" : editing ? "Create revision" : "Save workflow"}</button></footer></div></div>}
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
    <div className="modal-backdrop">
      <div className="modal settings-editor">
        <header><div><small>{profile.role} profile · {profile.engine}</small><h2>Edit profile</h2></div><button className="icon-button" onClick={onClose} aria-label="Close profile editor"><X /></button></header>
        <label>Profile name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label>Best used for<textarea rows={3} value={useCase} onChange={(event) => setUseCase(event.target.value)} placeholder="For example: programming, code review, and technical explanations" /><small>Auto mode compares this description with each prompt.</small></label>
        <label className="toggle-row"><span><strong>Default model</strong><small>Use this model when the chat selects Default.</small></span><input type="checkbox" checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} /></label>
        <div className="segmented compact">
          {(["basic", "advanced", "expert"] as Visibility[]).map((level) => <button key={level} className={visibility === level ? "active" : ""} onClick={() => setVisibility(level)}>{level}</button>)}
        </div>
        <div className="settings-list embedded">
          {fields.map((field) => {
            const target = field.scope === "load" ? loadSettings : requestSettings;
            return <div className="scoped-setting" key={`${field.scope}:${field.key}:${JSON.stringify(target[field.key])}`}><span className="scope-label">{field.scope}{field.restart_required ? " · restart required" : ""}</span><SettingControl field={field} value={target[field.key] ?? field.default} onChange={(value) => field.scope === "load" ? setLoadSettings({ ...loadSettings, [field.key]: value }) : setRequestSettings({ ...requestSettings, [field.key]: value })} /></div>;
          })}
          {!engine && <p className="muted">No capability schema is available for this profile engine.</p>}
        </div>
        {error && <div className="callout error">{error.message}</div>}
        <footer className="editor-actions"><button className="secondary danger" onClick={() => remove.mutate()} disabled={remove.isPending}>Delete profile</button><button className="secondary" onClick={() => reset.mutate()} disabled={reset.isPending}>Reset settings</button><button className="secondary" onClick={() => exportBundle.mutate()}>Export</button><button className="secondary" onClick={() => clone.mutate()}>Clone</button><button className="primary" onClick={() => save.mutate()} disabled={!name.trim() || save.isPending}>Save profile</button></footer>
      </div>
    </div>
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
    <div className="modal-backdrop">
      <div className="modal settings-editor">
        <header><div><small>{preset.role} generation preset</small><h2>Edit preset</h2></div><button className="icon-button" onClick={onClose} aria-label="Close preset editor"><X /></button></header>
        <label>Preset name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label className="toggle-row"><span><strong>Default {preset.role} preset</strong><small>Apply these values automatically to new {preset.role} generations.</small></span><input type="checkbox" checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} /></label>
        <div className="segmented compact">{(["basic", "advanced", "expert"] as Visibility[]).map((level) => <button key={level} className={visibility === level ? "active" : ""} onClick={() => setVisibility(level)}>{level}</button>)}</div>
        <div className="settings-list embedded">{fields.map((field) => <div className="scoped-setting" key={`${field.scope}:${field.key}:${JSON.stringify(settings[field.key])}`}><span className="scope-label">{field.scope}</span><SettingControl field={field} value={settings[field.key] ?? field.default} onChange={(value) => setSettings({ ...settings, [field.key]: value })} /></div>)}</div>
        {error && <div className="callout error">{error.message}</div>}
        <footer className="editor-actions"><button className="secondary danger" onClick={() => remove.mutate()}>Delete</button><button className="secondary" onClick={() => reset.mutate()}>Reset</button><button className="secondary" onClick={() => exportBundle.mutate()}>Export</button><button className="secondary" onClick={() => clone.mutate()}>Clone</button><button className="primary" onClick={() => save.mutate()} disabled={!name.trim() || save.isPending}>Save preset</button></footer>
      </div>
    </div>
  );
}

function SettingsView({ engines }: { engines: EngineCapabilities[] }) {
  const client = useQueryClient();
  const [hfToken, setHfToken] = useState("");
  const [selectedProfile, setSelectedProfile] = useState<ModelProfile | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<GenerationPreset | null>(null);
  const [presetName, setPresetName] = useState("");
  const [presetRole, setPresetRole] = useState<GenerationPreset["role"]>("chat");
  const [importError, setImportError] = useState("");
  const profileImport = useRef<HTMLInputElement>(null);
  const presetImport = useRef<HTMLInputElement>(null);
  const system = useQuery({ queryKey: ["system"], queryFn: api.system });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const workers = useQuery({ queryKey: ["workers"], queryFn: api.workers, refetchInterval: 3_000 });
  const backups = useQuery({ queryKey: ["backups"], queryFn: api.backups });
  const credential = useQuery({ queryKey: ["credential", "huggingface"], queryFn: api.credentialStatus });
  const saveCredential = useMutation({
    mutationFn: () => api.setHuggingFaceToken(hfToken),
    onSuccess: (value) => {
      setHfToken("");
      client.setQueryData(["credential", "huggingface"], value);
    },
  });
  const removeCredential = useMutation({
    mutationFn: api.deleteHuggingFaceToken,
    onSuccess: (value) => client.setQueryData(["credential", "huggingface"], value),
  });
  const refreshWorkers = () => void client.invalidateQueries({ queryKey: ["workers"] });
  const loadChat = useMutation({ mutationFn: api.loadChatWorker, onSuccess: refreshWorkers });
  const startMedia = useMutation({ mutationFn: api.startMediaWorker, onSuccess: refreshWorkers });
  const stopWorker = useMutation({ mutationFn: api.stopWorker, onSuccess: refreshWorkers });
  const createBackup = useMutation({ mutationFn: api.createBackup, onSuccess: () => void client.invalidateQueries({ queryKey: ["backups"] }) });
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
  return (
    <div className="page-view settings-page">
      <header className="page-header"><div><small>Settings</small><h1>Models, runtimes, and recovery</h1><p>Manage private model access, generation behavior, local engines, and backups.</p></div></header>
      <section>
        <div className="detail-title"><div><h2>Hugging Face access</h2><p>Private and gated model access is stored in your operating system credential vault. The token is never displayed after saving.</p></div><span className={`badge ${credential.data?.configured ? "tested" : ""}`}>{credential.data?.configured ? `Configured · ${credential.data.source.replace("credential_vault", "credential vault")}` : "Not configured"}</span></div>
        <div className="preset-create">
          <input aria-label="Hugging Face access token" type="password" autoComplete="off" placeholder="hf_…" value={hfToken} onChange={(event) => setHfToken(event.target.value)} disabled={credential.data?.source === "environment"} />
          <button className="primary" disabled={!hfToken.trim() || saveCredential.isPending || credential.data?.source === "environment" || credential.data?.vault_available === false} onClick={() => saveCredential.mutate()}>{saveCredential.isPending ? "Saving…" : "Save token"}</button>
          {credential.data?.configured && <button className="secondary danger" disabled={removeCredential.isPending || credential.data.source === "environment"} onClick={() => removeCredential.mutate()}>{removeCredential.isPending ? "Removing…" : "Remove"}</button>}
        </div>
        {credential.data?.source === "environment" && <p className="muted runtime-note">The LOCAL_LM_HF_TOKEN environment variable currently takes precedence. Unset it before managing the token here.</p>}
        {credential.data && !credential.data.vault_available && <div className="callout error">No supported operating-system credential vault is available. Configure one or use LOCAL_LM_HF_TOKEN for this process.</div>}
        {(credential.error || saveCredential.error || removeCredential.error) && <div className="callout error">{(credential.error || saveCredential.error || removeCredential.error)?.message}</div>}
      </section>
      <section><h2>Engines</h2><div className="engine-grid">{engines.map((engine) => <article className="engine-card" key={`${engine.engine}:${engine.roles.join()}`}><header><div className="model-icon"><Cpu /></div><div><h3>{engine.engine}</h3><p>{engine.roles.join(" · ")} · {engine.version}</p></div><StatusDot healthy={engine.healthy} /></header><div className="capability-list"><span>{engine.streaming ? "Streaming" : "Queued"}</span><span>{engine.tool_calling ? "Tool routing advertised" : "Workflow execution"}</span><span>{engine.settings.length} controls</span>{engine.roles.includes("chat") && <button className="secondary compact-button" onClick={() => toolProbe.mutate()} disabled={toolProbe.isPending}>{toolProbe.isPending ? "Testing…" : "Test structured tools"}</button>}</div></article>)}</div>{toolProbe.data && <div className={`callout ${toolProbe.data.passed ? "success" : "error"}`}>{toolProbe.data.passed ? `Structured tool schema passed on ${toolProbe.data.engine} ${toolProbe.data.version}.` : `Structured tool schema failed: ${toolProbe.data.error || "unknown response"}`}</div>}{toolProbe.error && <div className="callout error">{toolProbe.error.message}</div>}</section>
      <section><h2>Machine</h2>{system.data && <div className="metric-grid"><div className="cpu-metric"><Cpu /><span><strong>{system.data.cpu_model}</strong><small>CPU model</small></span></div><div><HardDrive /><span><strong>{formatBytes(system.data.disk_free_bytes)}</strong> disk free</span></div></div>}<div className="device-list">{system.data?.devices.filter((device) => device.kind !== "cpu").map((device) => <div key={device.id}><span className="device-icon"><Cpu size={18} /></span><span><strong>{device.name}</strong><small>{device.backend}</small></span></div>)}</div></section>
      <section>
        <div className="detail-title"><div><h2>Model profiles</h2><p>Store load-time and request-time controls independently for every model.</p></div><button className="secondary" onClick={() => profileImport.current?.click()}>Import profile</button></div>
        <input ref={profileImport} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0], "profile"); event.target.value = ""; }} />
        <div className="profile-table interactive">{profiles.data?.map((profile: ModelProfile) => <div key={profile.id}><span className="badge">{profile.role}</span><strong>{profile.is_default ? "Default" : profile.name}{profile.is_default ? " · default" : ""}</strong><span title={profile.use_case}>{profile.use_case || "No Auto use case yet"}</span><span className="row-actions">{profile.role === "chat" && profile.model_install_id && <button className="secondary compact-button" disabled={chatWorkerBusy || loadChat.isPending} title={chatWorkerBusy ? "Wait for active and queued chat jobs before changing the worker" : "Load this chat profile"} onClick={() => loadChat.mutate(profile.id)}>Load</button>}<button className="secondary compact-button" onClick={() => setSelectedProfile(profile)}>Edit</button></span></div>)}</div>
      </section>
      <section>
        <div className="detail-title"><div><h2>Generation presets</h2><p>Save reusable choices such as temperature, output length, dimensions, steps, and seed behavior.</p></div><button className="secondary" onClick={() => presetImport.current?.click()}>Import preset</button></div>
        <input ref={presetImport} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0], "preset"); event.target.value = ""; }} />
        <div className="preset-create"><input aria-label="New preset name" placeholder="New preset name" value={presetName} onChange={(event) => setPresetName(event.target.value)} /><select aria-label="New preset role" value={presetRole} onChange={(event) => setPresetRole(event.target.value as GenerationPreset["role"])}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option></select><button className="primary" disabled={!presetName.trim() || createPreset.isPending} onClick={() => createPreset.mutate()}><Plus size={15} />Create preset</button></div>
        <div className="profile-table interactive">{presets.data?.map((preset) => <div key={preset.id}><span className="badge">{preset.role}</span><strong>{preset.name}{preset.is_default ? " · default" : ""}</strong><span>{Object.keys(preset.settings_json).length} overrides</span><button className="secondary compact-button" onClick={() => setSelectedPreset(preset)}>Edit</button></div>)}</div>
        {(createPreset.error || importError) && <div className="callout error">{createPreset.error?.message || importError}</div>}
      </section>
      <section><div className="detail-title"><div><h2>Workers</h2><p>Configured workers start automatically when LM Atelier opens.</p></div></div><div className="engine-grid">{workers.data?.map((worker) => { const busy = worker.active_jobs + worker.queued_jobs > 0; const busyTitle = busy ? "Wait for active and queued jobs before changing this worker" : undefined; return <article className="engine-card" key={worker.name}><header><div><h3>{worker.name} worker</h3><p>{worker.state === "ready" ? `Ready · PID ${worker.pid}` : worker.state === "starting" ? "Starting and checking health" : worker.state === "exited" ? `Exited · code ${worker.exit_code ?? "unknown"}` : "Stopped or externally managed"}</p></div><StatusDot healthy={worker.state === "ready"} /></header><div className="worker-metrics"><span><strong>{worker.active_jobs}</strong> active</span><span><strong>{worker.queued_jobs}</strong> queued</span><span><strong>{formatBytes(worker.current_memory_bytes)}</strong> current RAM</span><span><strong>{formatBytes(worker.peak_memory_bytes)}</strong> measured peak</span>{worker.estimated_memory_bytes != null && <span><strong>{formatBytes(worker.estimated_memory_bytes)}</strong> estimated load</span>}</div><div className="capability-list">{worker.name === "media" && !worker.running && <button className="secondary compact-button" disabled={busy || startMedia.isPending} title={busyTitle} onClick={() => startMedia.mutate()}>Start ComfyUI</button>}{worker.running && <button className="secondary compact-button" disabled={busy || stopWorker.isPending} title={busyTitle} onClick={() => stopWorker.mutate(worker.name)}>Unload</button>}</div></article>; })}</div>{(loadChat.error || startMedia.error || stopWorker.error) && <div className="callout error">{(loadChat.error || startMedia.error || stopWorker.error)?.message}</div>}</section>
      <section><div className="detail-title"><div><h2>Recovery backups</h2><p>Keep 7 daily and 4 weekly verified snapshots. Media is optional so routine backups stay bounded.</p></div><div className="row-actions"><button className="secondary" onClick={() => createBackup.mutate(false)}>Back up state</button><button className="secondary" onClick={() => createBackup.mutate(true)}>Back up with media</button></div></div><div className="profile-table">{backups.data?.map((backup) => <div key={backup.name}><strong>{backup.name}</strong><span>{formatBytes(backup.size_bytes + backup.media_size_bytes)}</span><span>{backup.sha256.slice(0, 12)}</span><span>{backup.media_included ? "State + media" : backup.verified ? "Verified" : "State"}</span></div>)}</div></section>
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
  onDelete: () => void;
}) {
  const [title, setTitle] = useState(chat.title);
  const [projectId, setProjectId] = useState(chat.project_id ?? "");
  const [archived, setArchived] = useState(chat.archived);
  const [confirmUncertainMedia, setConfirmUncertainMedia] = useState(chat.confirm_uncertain_media);
  return <div className="modal-backdrop"><div className="modal workspace-editor"><header><div><small>Conversation</small><h2>Manage chat</h2></div><button className="icon-button" aria-label="Close chat manager" onClick={onClose}><X /></button></header><label>Title<input value={title} onChange={(event) => setTitle(event.target.value)} /></label><label>Project<select value={projectId} onChange={(event) => setProjectId(event.target.value)}><option value="">Unfiled</option>{projects.filter((project) => !project.archived).map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></label><label className="toggle-row"><span><strong>Confirm uncertain media</strong><small>Ask before Auto mode starts an image or video when the planner is unsure.</small></span><input type="checkbox" checked={confirmUncertainMedia} onChange={(event) => setConfirmUncertainMedia(event.target.checked)} /></label><label className="toggle-row"><span><strong>Archived</strong><small>Hide this chat from the active workspace without deleting its history.</small></span><input type="checkbox" checked={archived} onChange={(event) => setArchived(event.target.checked)} /></label><footer className="editor-actions"><button className="secondary danger" onClick={() => { if (window.confirm(`Delete ${chat.title} and its history?`)) onDelete(); }}>Delete chat</button><button className="secondary" onClick={onClose}>Cancel</button><button className="primary" disabled={!title.trim()} onClick={() => onSave({ title: title.trim(), project_id: projectId || null, archived, confirm_uncertain_media: confirmUncertainMedia })}>Save chat</button></footer></div></div>;
}

function ProjectManager({
  project,
  onClose,
  onSave,
  onDelete,
  onExport,
}: {
  project: Project;
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
  const workflows = useQuery({ queryKey: ["workflows"], queryFn: api.workflows });
  const workflowOptions = (kind: "image" | "video") => workflows.data?.filter((workflow) => workflow.operation.includes(kind)).flatMap((workflow) => workflow.revisions.map((revision) => <option key={revision.id} value={revision.id}>{workflow.name} · {workflow.operation.replaceAll("_", " ")} · v{revision.version}</option>));
  return <div className="modal-backdrop"><div className="modal workspace-editor"><header><div><small>Workspace</small><h2>Manage project</h2></div><button className="icon-button" aria-label="Close project manager" onClick={onClose}><X /></button></header><label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>Description<textarea rows={3} value={description} onChange={(event) => setDescription(event.target.value)} /></label><label>Project instructions<textarea rows={5} value={instructions} onChange={(event) => setInstructions(event.target.value)} /></label><label>Default image workflow revision<select value={imageWorkflowRevisionId} onChange={(event) => setImageWorkflowRevisionId(event.target.value)}><option value="">Use global default</option>{workflowOptions("image")}</select></label><label>Default video workflow revision<select value={videoWorkflowRevisionId} onChange={(event) => setVideoWorkflowRevisionId(event.target.value)}><option value="">Use global default</option>{workflowOptions("video")}</select></label><label className="toggle-row"><span><strong>Archived</strong><small>Hide this project while preserving its chats and media.</small></span><input type="checkbox" checked={archived} onChange={(event) => setArchived(event.target.checked)} /></label><div className="project-export-actions"><button className="secondary" onClick={() => onExport(false)}>Export metadata only</button><button className="secondary" onClick={() => onExport(true)}>Export with media</button></div><footer className="editor-actions"><button className="secondary danger" onClick={() => { if (window.confirm(`Delete ${project.name}? Its chats will become unfiled.`)) onDelete(); }}>Delete project</button><button className="secondary" onClick={onClose}>Cancel</button><button className="primary" disabled={!name.trim()} onClick={() => onSave({ name: name.trim(), description, instructions, archived, image_workflow_revision_id: imageWorkflowRevisionId || null, video_workflow_revision_id: videoWorkflowRevisionId || null })}>Save project</button></footer></div></div>;
}

function Sidebar({
  projects,
  chats,
  currentChatId,
  view,
  onChat,
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
  currentChatId: string | null;
  view: View;
  onChat: (id: string) => void;
  onView: (view: View) => void;
  onNewChat: (projectId?: string | null) => void;
  onNewProject: () => void;
  onExportProject: (id: string, includeMedia?: boolean) => void;
  onImportProject: (file: File) => void;
  onUpdateChat: (id: string, values: Partial<Chat>) => void;
  onDeleteChat: (id: string) => void;
  onUpdateProject: (id: string, values: Partial<Project>) => void;
  onDeleteProject: (id: string) => void;
}) {
  const [openProjects, setOpenProjects] = useState<Set<string>>(new Set(projects.map((project) => project.id)));
  const [search, setSearch] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [managedChat, setManagedChat] = useState<Chat | null>(null);
  const [managedProject, setManagedProject] = useState<Project | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const projectImport = useRef<HTMLInputElement>(null);
  useEffect(() => {
    setOpenProjects((current) => new Set([...current, ...projects.map((project) => project.id)]));
  }, [projects]);
  const normalizedSearch = search.trim().toLowerCase();
  const visibleChats = chats.filter((chat) => (showArchived || !chat.archived) && (!normalizedSearch || chat.title.toLowerCase().includes(normalizedSearch)));
  const visibleProjects = projects.filter((project) => (showArchived || !project.archived) && (!normalizedSearch || project.name.toLowerCase().includes(normalizedSearch) || visibleChats.some((chat) => chat.project_id === project.id)));
  const unfiled = visibleChats.filter((chat) => !chat.project_id);
  const chatRow = (chat: Chat) => <div className="sidebar-chat-row" key={chat.id}><button className={`chat-main ${view === "chat" && currentChatId === chat.id ? "active" : ""}`} onClick={() => { onChat(chat.id); setMobileOpen(false); }}><MessageSquare size={14} /><span>{chat.title}</span>{chat.archived && <small>Archived</small>}</button><button className="inline-add" aria-label={`Manage ${chat.title}`} onClick={() => setManagedChat(chat)}><MoreHorizontal size={13} /></button></div>;
  return (
    <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`}>
      <div className="brand"><div className="brand-mark"><AtelierMark /></div><span>LM Atelier<small>Local creative studio</small></span><button className="icon-button mobile-menu" aria-label="Toggle navigation" aria-expanded={mobileOpen} onClick={() => setMobileOpen((open) => !open)}><Menu /></button></div>
      <button className="new-chat" onClick={() => { onNewChat(null); setMobileOpen(false); }}><Plus size={18} />New chat</button>
      <nav className="primary-nav"><button className={view === "media" ? "active" : ""} onClick={() => { onView("media"); setMobileOpen(false); }}><ImageIcon />Media library</button><button className={view === "models" ? "active" : ""} onClick={() => { onView("models"); setMobileOpen(false); }}><Library />Model library</button><button className={view === "workflows" ? "active" : ""} onClick={() => { onView("workflows"); setMobileOpen(false); }}><WorkflowIcon />Workflows</button></nav>
      <div className="workspace-search"><Search size={14} /><input aria-label="Search projects and chats" placeholder="Search workspace" value={search} onChange={(event) => setSearch(event.target.value)} /><button className={showArchived ? "active" : ""} aria-pressed={showArchived} onClick={() => setShowArchived((value) => !value)}>Archived</button></div>
      <div className="workspace-tree" role="region" aria-label="Projects and chats">
        <div className="sidebar-section">
          <div className="section-title"><span>Projects</span><input ref={projectImport} hidden type="file" accept=".zip,.lm-atelier.zip,application/zip" onChange={(event) => { const file = event.target.files?.[0]; if (file) onImportProject(file); event.target.value = ""; }} /><button aria-label="Import project" onClick={() => projectImport.current?.click()}><Upload size={14} /></button><button aria-label="New project" onClick={onNewProject}><Plus size={15} /></button></div>
          {visibleProjects.map((project) => {
            const open = openProjects.has(project.id);
            const projectMatches = normalizedSearch && project.name.toLowerCase().includes(normalizedSearch);
            const projectChats = chats.filter((chat) => chat.project_id === project.id && (showArchived || !chat.archived) && (!normalizedSearch || projectMatches || chat.title.toLowerCase().includes(normalizedSearch)));
            return (
              <div className="project-group" key={project.id}>
                <div className="project-row">
                  <button className="project-main" onClick={() => setOpenProjects((current) => {
                    const next = new Set(current);
                    if (open) next.delete(project.id);
                    else next.add(project.id);
                    return next;
                  })}>
                    <ChevronDown className={open ? "" : "closed"} size={14} />
                    <Folder size={16} />
                    <span>{project.name}</span>
                  </button>
                  <button className="inline-add" onClick={() => onNewChat(project.id)} aria-label={`New chat in ${project.name}`}><Plus size={13} /></button>
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
      <div className="sidebar-footer"><button className={view === "settings" ? "active" : ""} onClick={() => { onView("settings"); setMobileOpen(false); }}><Settings />Settings</button></div>
      {managedChat && <ChatManager chat={managedChat} projects={projects} onClose={() => setManagedChat(null)} onSave={(values) => { onUpdateChat(managedChat.id, values); setManagedChat(null); }} onDelete={() => { onDeleteChat(managedChat.id); setManagedChat(null); }} />}
      {managedProject && <ProjectManager project={managedProject} onClose={() => setManagedProject(null)} onSave={(values) => { onUpdateProject(managedProject.id, values); setManagedProject(null); }} onDelete={() => { onDeleteProject(managedProject.id); setManagedProject(null); }} onExport={(includeMedia) => onExportProject(managedProject.id, includeMedia)} />}
    </aside>
  );
}

function JobsPanel() {
  const client = useQueryClient();
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3_000 });
  const refresh = () => void client.invalidateQueries({ queryKey: ["jobs"] });
  const cancel = useMutation({ mutationFn: api.cancelJob, onSuccess: refresh });
  const pause = useMutation({ mutationFn: api.pauseDownload, onSuccess: refresh });
  const resume = useMutation({ mutationFn: api.resumeDownload, onSuccess: refresh });
  const active = jobs.data?.filter((job) => ["queued", "running", "paused"].includes(job.status)) ?? [];
  if (!active.length) return null;
  return <div className="jobs-panel"><header><Activity size={16} /><span>{active.length} active job{active.length === 1 ? "" : "s"}</span></header>{active.map((job) => <div className="job-row" key={job.id}><div><strong>{job.kind}</strong><small>{job.phase}</small><div className="progress-track"><div style={{ width: `${Math.max(4, job.progress * 100)}%` }} /></div></div><span className="job-actions">{job.kind === "download" && (job.status === "paused" ? <button className="icon-button" aria-label="Resume download" onClick={() => resume.mutate(job.id)}><Play size={16} /></button> : <button className="icon-button" aria-label="Pause download" onClick={() => pause.mutate(job.id)}><Pause size={16} /></button>)}<button className="icon-button" aria-label="Cancel job" onClick={() => cancel.mutate(job.id)}><CircleStop size={17} /></button></span></div>)}</div>;
}

export default function App() {
  const client = useQueryClient();
  const [view, setView] = useState<View>("chat");
  const [currentChatId, setCurrentChatId] = useState<string | null>(() => localStorage.getItem("local-lm-chat"));
  const [liveText, setLiveText] = useState<Record<string, string>>({});
  const [turnSettingsByChat, setTurnSettingsByChat] = useState<Record<string, Record<string, unknown>>>({});
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects(true) });
  const chats = useQuery({ queryKey: ["chats"], queryFn: () => api.chats(null, true) });
  const chat = useQuery({ queryKey: ["chat", currentChatId], queryFn: () => api.chat(currentChatId!), enabled: Boolean(currentChatId) });
  const engines = useQuery({ queryKey: ["engines"], queryFn: api.engines });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const workflows = useQuery({ queryKey: ["workflows"], queryFn: api.workflows });

  useEffect(() => {
    let dispose: (() => void) | undefined;
    let mediaRefresh: number | undefined;
    const scheduleMediaRefresh = () => {
      if (mediaRefresh !== undefined) return;
      mediaRefresh = window.setTimeout(() => {
        mediaRefresh = undefined;
        void client.invalidateQueries({ queryKey: ["chat"] });
      }, 100);
    };
    void connectEvents(
      (event: AppEvent) => {
        if (event.type === "text.delta") {
          const messageId = String(event.payload.assistant_message_id ?? "");
          const text = String(event.payload.text ?? "");
          if (messageId) setLiveText((current) => ({ ...current, [messageId]: `${current[messageId] ?? ""}${text}` }));
          return;
        }
        if (event.type.includes("progress") || event.type.startsWith("download.")) void client.invalidateQueries({ queryKey: ["jobs"] });
        if (event.type === "download.completed") {
          void client.invalidateQueries({ queryKey: ["models"] });
          void client.invalidateQueries({ queryKey: ["profiles"] });
          void client.invalidateQueries({ queryKey: ["model-storage"] });
        }
        if (["generation.progress", "generation.preview"].includes(event.type)) {
          scheduleMediaRefresh();
        }
        if (["run.completed", "run.failed", "run.cancelled"].includes(event.type)) {
          if (mediaRefresh !== undefined) window.clearTimeout(mediaRefresh);
          mediaRefresh = undefined;
          void client.invalidateQueries({ queryKey: ["chat"] });
          void client.invalidateQueries({ queryKey: ["chats"] });
          void client.invalidateQueries({ queryKey: ["jobs"] });
          void client.invalidateQueries({ queryKey: ["artifacts"] });
          void client.invalidateQueries({ queryKey: ["artifact-storage"] });
          window.setTimeout(() => setLiveText({}), 200);
        }
      },
      () => undefined,
    ).then((cleanup) => { dispose = cleanup; });
    return () => {
      if (mediaRefresh !== undefined) window.clearTimeout(mediaRefresh);
      dispose?.();
    };
  }, [client]);

  const createChat = useMutation({
    mutationFn: (projectId?: string | null) => api.createChat(projectId),
    onSuccess: (created) => { setCurrentChatId(created.id); localStorage.setItem("local-lm-chat", created.id); setView("chat"); void client.invalidateQueries({ queryKey: ["chats"] }); },
  });
  const createProject = useMutation({
    mutationFn: api.createProject,
    onSuccess: () => void client.invalidateQueries({ queryKey: ["projects"] }),
  });
  const send = useMutation({
    mutationFn: ({ text, mode, artifacts, settings }: { text: string; mode: RoutingMode; artifacts: string[]; settings: Record<string, unknown> }) => api.sendTurn(currentChatId!, text, mode, artifacts, settings),
    onSuccess: (accepted) => {
      client.setQueryData<ChatDetail>(["chat", currentChatId], (current) => current ? { ...current, active_head_message_id: accepted.assistant_message.id, messages: [...current.messages, accepted.user_message, accepted.assistant_message] } : current);
      void client.invalidateQueries({ queryKey: ["chats"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const regenerate = useMutation({
    mutationFn: ({ messageId, settings }: { messageId: string; settings: Record<string, unknown> }) => api.regenerateMessage(messageId, settings),
    onSuccess: (accepted) => {
      client.setQueryData<ChatDetail>(["chat", currentChatId], (current) => current ? { ...current, active_head_message_id: accepted.assistant_message.id, messages: [...current.messages, accepted.user_message, accepted.assistant_message] } : current);
      void client.invalidateQueries({ queryKey: ["chats"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const branch = useMutation({
    mutationFn: ({ messageId, text, settings }: { messageId: string; text: string; settings: Record<string, unknown> }) => api.branchMessage(messageId, text, settings),
    onSuccess: (accepted) => {
      client.setQueryData<ChatDetail>(["chat", currentChatId], (current) => current ? { ...current, active_head_message_id: accepted.assistant_message.id, messages: [...current.messages, accepted.user_message, accepted.assistant_message] } : current);
      void client.invalidateQueries({ queryKey: ["chats"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const stop = useMutation({
    mutationFn: () => api.cancelChat(currentChatId!),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["chat", currentChatId] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const updateChat = useMutation({
    mutationFn: (values: Partial<Chat>) => api.updateChat(currentChatId!, values),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["chat", currentChatId] });
      void client.invalidateQueries({ queryKey: ["chats"] });
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
    mutationFn: api.deleteChat,
    onSuccess: (_value, deletedId) => {
      setTurnSettingsByChat((current) => {
        const next = { ...current };
        delete next[deletedId];
        return next;
      });
      if (currentChatId === deletedId) {
        setCurrentChatId(null);
        localStorage.removeItem("local-lm-chat");
      }
      void client.invalidateQueries({ queryKey: ["chats"] });
    },
  });
  const updateProject = useMutation({
    mutationFn: ({ id, values }: { id: string; values: Partial<Project> }) => api.updateProject(id, values),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["projects"] }),
  });
  const deleteProject = useMutation({
    mutationFn: api.deleteProject,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["projects"] });
      void client.invalidateQueries({ queryKey: ["chats"] });
    },
  });
  const exportProject = useMutation({
    mutationFn: ({ id, includeMedia = true }: { id: string; includeMedia?: boolean }) => api.exportProject(id, includeMedia),
    onSuccess: (artifact) => {
      const link = document.createElement("a");
      link.href = artifact.url;
      link.download = "";
      link.click();
    },
  });
  const importProject = useMutation({
    mutationFn: api.importProject,
    onSuccess: (project) => {
      void client.invalidateQueries({ queryKey: ["projects"] });
      void client.invalidateQueries({ queryKey: ["chats"] });
      window.setTimeout(() => {
        const importedChat = client.getQueryData<Chat[]>(["chats"])?.find((item) => item.project_id === project.id);
        if (importedChat) {
          setCurrentChatId(importedChat.id);
          localStorage.setItem("local-lm-chat", importedChat.id);
          setView("chat");
        }
      }, 100);
    },
  });

  useEffect(() => {
    if (!currentChatId) {
      const firstActive = chats.data?.find((candidate) => !candidate.archived);
      if (firstActive) setCurrentChatId(firstActive.id);
    }
  }, [chats.data, currentChatId]);

  const allChats = useMemo(() => chats.data ?? [], [chats.data]);
  const allProjects = useMemo(() => projects.data ?? [], [projects.data]);
  const activeContent = useMemo(() => {
    if (view === "media") return <MediaLibraryView />;
    if (view === "models") return <ModelsView />;
    if (view === "workflows") return <WorkflowsView />;
    if (view === "settings") return <SettingsView engines={engines.data ?? []} />;
    return <ChatView chat={chat.data} engines={engines.data ?? []} profiles={profiles.data ?? []} workflows={workflows.data ?? []} project={allProjects.find((item) => item.id === chat.data?.project_id)} liveText={liveText} settings={currentChatId ? turnSettingsByChat[currentChatId] ?? {} : {}} onSettings={(settings) => {
      if (currentChatId) setTurnSettingsByChat((current) => ({ ...current, [currentChatId]: settings }));
    }} onProfile={(field, id) => updateChat.mutate({ [field]: id })} onRegenerate={(messageId, settings) => regenerate.mutate({ messageId, settings })} onEdit={(messageId, text, settings) => branch.mutate({ messageId, text, settings })} onStop={() => stop.mutate()} onSend={(text, mode, artifacts, settings) => send.mutate({ text, mode, artifacts, settings })} />;
  }, [view, engines.data, profiles.data, workflows.data, allProjects, chat.data, liveText, currentChatId, turnSettingsByChat, send, regenerate, branch, stop, updateChat]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <Sidebar projects={allProjects} chats={allChats} currentChatId={currentChatId} view={view} onChat={(id) => { setCurrentChatId(id); localStorage.setItem("local-lm-chat", id); setView("chat"); }} onView={setView} onNewChat={(projectId) => createChat.mutate(projectId)} onNewProject={() => { const name = window.prompt("Project name"); if (name?.trim()) createProject.mutate(name.trim()); }} onExportProject={(id, includeMedia) => exportProject.mutate({ id, includeMedia })} onImportProject={(file) => importProject.mutate(file)} onUpdateChat={(id, values) => manageChat.mutate({ id, values })} onDeleteChat={(id) => deleteChat.mutate(id)} onUpdateProject={(id, values) => updateProject.mutate({ id, values })} onDeleteProject={(id) => deleteProject.mutate(id)} />
      <main id="main-content" tabIndex={-1}>{activeContent}</main>
      <JobsPanel />
      {(send.error || regenerate.error || branch.error || stop.error || createChat.error || createProject.error || importProject.error || manageChat.error || deleteChat.error || updateProject.error || deleteProject.error) && <div className="toast error"><X size={16} />{(send.error || regenerate.error || branch.error || stop.error || createChat.error || createProject.error || importProject.error || manageChat.error || deleteChat.error || updateProject.error || deleteProject.error)?.message}</div>}
    </div>
  );
}
