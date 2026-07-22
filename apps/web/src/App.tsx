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
  Plus,
  RotateCcw,
  Search,
  Send,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Workflow as WorkflowIcon,
  X,
} from "lucide-react";
import { api, connectEvents } from "./api";
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
} from "./types";

type View = "chat" | "models" | "workflows" | "settings";
type Visibility = "basic" | "advanced" | "expert";

const visibilityRank: Record<Visibility, number> = { basic: 0, advanced: 1, expert: 2 };

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

function StatusDot({ healthy }: { healthy: boolean }) {
  return <span className={`status-dot ${healthy ? "healthy" : "offline"}`} />;
}

function ArtifactPart({ part }: { part: MessagePart }) {
  if (!part.artifact_id) return null;
  const source = `/api/artifacts/${encodeURIComponent(part.artifact_id)}/content`;
  if (part.type === "image") {
    return (
      <figure className="media-card">
        <img src={source} alt="Generated result" loading="lazy" />
        <figcaption>
          <ImageIcon size={14} /> Generated image
        </figcaption>
      </figure>
    );
  }
  return (
    <figure className="media-card">
      <video src={source} controls preload="metadata" />
      <figcaption>
        <Film size={14} /> Generated video
      </figcaption>
    </figure>
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
        {message.role === "assistant" && message.status === "complete" && (
          <div className="message-meta">
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
  if (field.type === "boolean") {
    return (
      <label className="setting-row toggle-row">
        <span><strong>{field.label}</strong><small>{field.help}</small></span>
        <input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
      </label>
    );
  }
  if (field.type === "enum") {
    return (
      <label className="setting-row">
        <span><strong>{field.label}</strong><small>{field.help}</small></span>
        <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>
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
      <input value={String(value ?? "")} onChange={(event) => onChange(event.target.value)} />
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
}: {
  open: boolean;
  onClose: () => void;
  mode: RoutingMode;
  engines: EngineCapabilities[];
  values: Record<string, unknown>;
  onValues: (values: Record<string, unknown>) => void;
}) {
  const [visibility, setVisibility] = useState<Visibility>("basic");
  const role = mode === "video" ? "video" : mode === "image" ? "image" : "chat";
  const engine = engines.find((item) => item.roles.includes(role));
  const fields = (engine?.settings ?? []).filter(
    (field) => visibilityRank[field.visibility] <= visibilityRank[visibility] && field.available,
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
  onSend,
  onStop,
}: {
  chat: Chat;
  engines: EngineCapabilities[];
  busy: boolean;
  onSend: (text: string, mode: RoutingMode, artifacts: string[], settings: Record<string, unknown>) => void;
  onStop: () => void;
}) {
  const [text, setText] = useState("");
  const [mode, setMode] = useState<RoutingMode>(chat.routing_mode);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [attachments, setAttachments] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const submit = () => {
    if (!text.trim() || busy) return;
    onSend(text.trim(), mode, attachments, settings);
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
      <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} mode={mode} engines={engines} values={settings} onValues={setSettings} />
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

function ChatView({
  chat,
  engines,
  profiles,
  liveText,
  onSend,
  onProfile,
  onRegenerate,
  onEdit,
  onStop,
}: {
  chat?: ChatDetail;
  engines: EngineCapabilities[];
  profiles: ModelProfile[];
  liveText: Record<string, string>;
  onSend: (text: string, mode: RoutingMode, artifacts: string[], settings: Record<string, unknown>) => void;
  onProfile: (field: "active_chat_profile_id" | "active_image_profile_id" | "active_video_profile_id", id: string | null) => void;
  onRegenerate: (messageId: string) => void;
  onEdit: (messageId: string, text: string) => void;
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
            return <label key={role}><span>{role}</span><select value={chat[field] ?? ""} onChange={(event) => onProfile(field, event.target.value || null)}><option value="">Default</option>{profiles.filter((profile) => profile.role === role).map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}</select></label>;
          })}
          <button className="icon-button"><MoreHorizontal /></button>
        </div>
      </div>
      <div className="messages">
        {messages.length === 0 ? (
          <EmptyState icon={<Sparkles />} title="What should we make?" body="Ask a question or describe an image or video. Auto mode chooses the appropriate local model." />
        ) : messages.map((message) => <MessageBubble key={message.id} message={message} liveText={liveText[message.id]} onRegenerate={busy ? undefined : onRegenerate} onEdit={busy ? undefined : onEdit} />)}
        <div ref={endRef} />
      </div>
      <Composer chat={chat} engines={engines} busy={busy} onSend={onSend} onStop={onStop} />
    </div>
  );
}

function ModelCard({ model, role, onDownload, pending }: { model: CatalogModel; role: string; onDownload: () => void; pending: boolean }) {
  return (
    <article className="model-card">
      <div className="model-icon">{role === "video" ? <Film /> : role === "image" ? <ImageIcon /> : <Bot />}</div>
      <div className="model-copy">
        <h3>{model.name}</h3><p>{model.author} · {model.pipeline_tag || model.library_name || "model"}</p>
        <div className="badges"><span className={`badge ${model.compatibility}`}>{model.compatibility.replace("_", " ")}</span>{model.gated && <span className="badge">Gated</span>}{model.formats.slice(0, 2).map((format) => <span className="badge" key={format}>{format}</span>)}{model.quantizations.slice(0, 2).map((value) => <span className="badge" key={value}>{value}</span>)}</div>
        <small>{model.compatibility_reasons.join(" · ")}</small>
      </div>
      <div className="model-stats"><span><Download size={14} />{model.downloads?.toLocaleString() ?? "—"}</span><button className="primary compact-button" onClick={onDownload} disabled={pending}>{pending ? "Queued" : "Choose files"}</button></div>
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
  const [detailModel, setDetailModel] = useState<CatalogModel | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const catalogFilters = { compatibility, file_format: fileFormat, gated, quantization, license_id: licenseId, architecture };
  const catalog = useInfiniteQuery({
    queryKey: ["catalog", submitted, role, sort, catalogFilters],
    queryFn: ({ pageParam }) => api.catalog(submitted, role, sort, pageParam, catalogFilters),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });
  const catalogItems = useMemo(() => catalog.data?.pages.flatMap((page) => page.items) ?? [], [catalog.data]);
  const detail = useQuery({ queryKey: ["catalog-detail", detailModel?.remote_id], queryFn: () => api.catalogDetail(detailModel!.remote_id), enabled: Boolean(detailModel) });
  const recipes = useQuery({ queryKey: ["recipes"], queryFn: api.recipes });
  const installed = useQuery({ queryKey: ["models"], queryFn: api.models });
  const storage = useQuery({ queryKey: ["model-storage"], queryFn: api.modelStorage });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const download = useMutation({
    mutationFn: ({ remoteId, files }: { remoteId: string; files: string[] }) => api.download(remoteId, role, role === "chat" ? "llama.cpp" : "comfyui", files),
    onSuccess: () => { setDetailModel(null); setSelectedFiles([]); void client.invalidateQueries({ queryKey: ["jobs"] }); },
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
    mutationFn: api.deleteModel,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const cleanupDownloads = useMutation({
    mutationFn: api.cleanupDownloads,
    onSuccess: () => void client.invalidateQueries({ queryKey: ["model-storage"] }),
  });
  return (
    <div className="page-view">
      <header className="page-header"><div><small>Model library</small><h1>Find the right local model</h1><p>Search Hugging Face, inspect compatibility, and manage downloads without leaving the app.</p></div><div className="storage-actions"><div className="storage-pill"><HardDrive size={17} />{storage.data?.installed_count ?? installed.data?.length ?? 0} installed · {formatBytes(storage.data?.installed_bytes)}</div><button className="secondary compact-button" disabled={!storage.data?.partial_download_count || cleanupDownloads.isPending} onClick={() => cleanupDownloads.mutate()}>Clean {storage.data?.partial_download_count ?? 0} partial</button></div></header>
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
      <div className="catalog-filters"><select aria-label="Compatibility filter" value={compatibility} onChange={(event) => setCompatibility(event.target.value)}><option value="">All compatibility</option><option value="likely">Likely compatible</option><option value="advanced_import">Advanced import</option><option value="unsupported">Unsupported</option></select><select aria-label="Format filter" value={fileFormat} onChange={(event) => setFileFormat(event.target.value)}><option value="">All formats</option><option value="gguf">GGUF</option><option value="safetensors">safetensors</option></select><select aria-label="Access filter" value={gated} onChange={(event) => setGated(event.target.value)}><option value="">All access</option><option value="open">Open</option><option value="gated">Gated</option></select><input aria-label="Quantization filter" placeholder="Quantization (Q4_K_M, FP8…)" value={quantization} onChange={(event) => setQuantization(event.target.value)} /><input aria-label="Architecture filter" placeholder="Architecture" value={architecture} onChange={(event) => setArchitecture(event.target.value)} /><input aria-label="License filter" placeholder="License" value={licenseId} onChange={(event) => setLicenseId(event.target.value)} /></div>
      {(installed.data?.length ?? 0) > 0 && <section><h2>Installed models</h2><div className="profile-table model-installs">{installed.data?.map((model) => { const bound = profiles.data?.some((profile) => profile.model_install_id === model.id) ?? false; return <div key={model.id}><span className="badge">{model.role}</span><strong>{model.name}</strong><span>{formatBytes(model.size_bytes)}</span><span className="row-actions"><button className="secondary compact-button" disabled={bound || createProfile.isPending} onClick={() => createProfile.mutate(model)}>{bound ? "Profile ready" : "Create profile"}</button><button className="secondary compact-button danger" disabled={bound || deleteModel.isPending} title={bound ? "Delete the model profile first" : "Delete installed model"} onClick={() => { if (window.confirm(`Delete ${model.name} from local storage?`)) deleteModel.mutate(model.id); }}>Delete</button></span></div>; })}</div></section>}
      {(deleteModel.error || cleanupDownloads.error) && <div className="callout error">{deleteModel.error?.message || cleanupDownloads.error?.message}</div>}
      {catalog.isLoading && <div className="loading-line" />}
      {catalog.error && <div className="callout error">{catalog.error.message}</div>}
      <div className="model-grid">{catalogItems.map((model) => <ModelCard key={model.remote_id} model={model} role={role} pending={download.isPending && download.variables?.remoteId === model.remote_id} onDownload={() => { setDetailModel(model); setSelectedFiles([]); }} />)}</div>
      {catalog.hasNextPage && <div className="load-more"><button className="secondary" disabled={catalog.isFetchingNextPage} onClick={() => void catalog.fetchNextPage()}>{catalog.isFetchingNextPage ? "Loading…" : "Load more models"}</button></div>}
      {detailModel && <div className="modal-backdrop"><div className="modal model-files"><header><div><small>Advanced install</small><h2>{detailModel.remote_id}</h2></div><button className="icon-button" onClick={() => setDetailModel(null)}><X /></button></header><p>Select exact files. Chat installs may leave this empty to choose the smallest GGUF automatically; image and video installs require an explicit selection.</p><div className="file-picker">{detail.data?.files.map((file) => <label key={file.filename}><input type="checkbox" checked={selectedFiles.includes(file.filename)} onChange={(event) => setSelectedFiles((current) => event.target.checked ? [...current, file.filename] : current.filter((name) => name !== file.filename))} /><span><strong>{file.filename}</strong><small>{formatBytes(file.size)}</small></span></label>)}</div>{detail.error && <div className="callout error">{detail.error.message}</div>}<footer><span>{formatBytes(detail.data?.files.filter((file) => selectedFiles.includes(file.filename)).reduce((total, file) => total + (file.size ?? 0), 0))} selected</span><button className="primary" disabled={detail.isLoading || (role !== "chat" && selectedFiles.length === 0)} onClick={() => download.mutate({ remoteId: detailModel.remote_id, files: selectedFiles })}>Queue download</button></footer></div></div>}
    </div>
  );
}

function WorkflowsView() {
  const client = useQueryClient();
  const workflows = useQuery({ queryKey: ["workflows"], queryFn: api.workflows });
  const [selected, setSelected] = useState<Workflow | null>(null);
  const [newOpen, setNewOpen] = useState(false);
  const [name, setName] = useState("Custom image workflow");
  const [operation, setOperation] = useState("text_to_image");
  const [graph, setGraph] = useState("{}");
  const [inputSchema, setInputSchema] = useState("{}");
  const [trusted, setTrusted] = useState(false);
  const create = useMutation({
    mutationFn: () => api.createWorkflow({ name, operation, engine: "comfyui", api_graph: JSON.parse(graph), ui_graph: {}, input_schema: JSON.parse(inputSchema), dependencies: {}, trusted }),
    onSuccess: () => { setNewOpen(false); void client.invalidateQueries({ queryKey: ["workflows"] }); },
  });
  const validate = useMutation({ mutationFn: (id: string) => api.validateWorkflow(id) });
  return (
    <div className="page-view">
      <header className="page-header"><div><small>Workflow studio</small><h1>Media pipelines</h1><p>Version complete ComfyUI graphs and expose their inputs as reusable generation controls.</p></div><button className="primary" onClick={() => setNewOpen(true)}><Plus size={17} />Import workflow</button></header>
      <div className="workflow-layout">
        <div className="workflow-list">{workflows.data?.map((workflow) => <button key={workflow.id} className={selected?.id === workflow.id ? "selected" : ""} onClick={() => setSelected(workflow)}><WorkflowIcon size={18} /><span><strong>{workflow.name}</strong><small>{workflow.operation} · {workflow.revisions.length} revision{workflow.revisions.length === 1 ? "" : "s"}</small></span></button>)}</div>
        <div className="workflow-detail">{selected ? <><div className="detail-title"><div><small>{selected.operation}</small><h2>{selected.name}</h2></div><button className="secondary" onClick={() => validate.mutate(selected.id)}>Validate</button></div><pre>{JSON.stringify(selected.revisions.find((revision) => revision.id === selected.current_revision_id)?.api_graph_json ?? {}, null, 2)}</pre>{validate.data && <div className={`callout ${validate.data.valid ? "success" : "error"}`}>{validate.data.valid ? "Workflow is valid for the active media engine." : validate.data.errors.join("\n")}</div>}</> : <EmptyState icon={<WorkflowIcon />} title="Select a workflow" body="Inspect its pinned revision, inputs, dependencies, and validation state." />}</div>
      </div>
      {newOpen && <div className="modal-backdrop"><div className="modal"><header><h2>Import ComfyUI API workflow</h2><button className="icon-button" onClick={() => setNewOpen(false)}><X /></button></header><label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>Operation<select value={operation} onChange={(event) => setOperation(event.target.value)}><option value="text_to_image">Text to image</option><option value="image_to_image">Image to image</option><option value="text_to_video">Text to video</option><option value="image_to_video">Image to video</option></select></label><label>API-format workflow JSON<textarea rows={10} value={graph} onChange={(event) => setGraph(event.target.value)} /></label><label>Declared input schema JSON<textarea rows={5} value={inputSchema} onChange={(event) => setInputSchema(event.target.value)} /></label><label className="toggle-row"><span><strong>Trust this workflow</strong><small>Only enable after reviewing every node and dependency.</small></span><input type="checkbox" checked={trusted} onChange={(event) => setTrusted(event.target.checked)} /></label>{create.error && <div className="callout error">{create.error.message}</div>}<footer><button className="secondary" onClick={() => setNewOpen(false)}>Cancel</button><button className="primary" onClick={() => create.mutate()}>Save workflow</button></footer></div></div>}
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
  const [isDefault, setIsDefault] = useState(profile.is_default);
  const [loadSettings, setLoadSettings] = useState(profile.load_settings_json);
  const [requestSettings, setRequestSettings] = useState(profile.request_settings_json);
  const [visibility, setVisibility] = useState<Visibility>("basic");
  const refresh = () => void client.invalidateQueries({ queryKey: ["profiles"] });
  const save = useMutation({
    mutationFn: () => api.updateProfile(profile.id, {
      name,
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
  const fields = (engine?.settings ?? []).filter(
    (field) => visibilityRank[field.visibility] <= visibilityRank[visibility] && field.available,
  );
  const error = save.error ?? clone.error ?? reset.error ?? remove.error ?? exportBundle.error;
  return (
    <div className="modal-backdrop">
      <div className="modal settings-editor">
        <header><div><small>{profile.role} profile · {profile.engine}</small><h2>Edit profile</h2></div><button className="icon-button" onClick={onClose} aria-label="Close profile editor"><X /></button></header>
        <label>Profile name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label className="toggle-row"><span><strong>Default {profile.role} profile</strong><small>Use this profile when a chat does not select another one.</small></span><input type="checkbox" checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} /></label>
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
  const fields = (engine?.settings ?? []).filter((field) => visibilityRank[field.visibility] <= visibilityRank[visibility] && field.available);
  const error = save.error ?? clone.error ?? reset.error ?? remove.error ?? exportBundle.error;
  return (
    <div className="modal-backdrop">
      <div className="modal settings-editor">
        <header><div><small>{preset.role} generation preset</small><h2>Edit preset</h2></div><button className="icon-button" onClick={onClose} aria-label="Close preset editor"><X /></button></header>
        <label>Preset name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label className="toggle-row"><span><strong>Default {preset.role} preset</strong><small>Apply these values when no explicit preset is selected.</small></span><input type="checkbox" checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} /></label>
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
  const [selectedProfile, setSelectedProfile] = useState<ModelProfile | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<GenerationPreset | null>(null);
  const [presetName, setPresetName] = useState("");
  const [presetRole, setPresetRole] = useState<GenerationPreset["role"]>("chat");
  const [importError, setImportError] = useState("");
  const profileImport = useRef<HTMLInputElement>(null);
  const presetImport = useRef<HTMLInputElement>(null);
  const system = useQuery({ queryKey: ["system"], queryFn: api.system });
  const platforms = useQuery({ queryKey: ["platforms"], queryFn: api.platforms });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const workers = useQuery({ queryKey: ["workers"], queryFn: api.workers, refetchInterval: 3_000 });
  const backups = useQuery({ queryKey: ["backups"], queryFn: api.backups });
  const refreshWorkers = () => void client.invalidateQueries({ queryKey: ["workers"] });
  const loadChat = useMutation({ mutationFn: api.loadChatWorker, onSuccess: refreshWorkers });
  const startMedia = useMutation({ mutationFn: api.startMediaWorker, onSuccess: refreshWorkers });
  const stopWorker = useMutation({ mutationFn: api.stopWorker, onSuccess: refreshWorkers });
  const createBackup = useMutation({ mutationFn: api.createBackup, onSuccess: () => void client.invalidateQueries({ queryKey: ["backups"] }) });
  const toolProbe = useMutation({ mutationFn: api.probeChatTools });
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
      <header className="page-header"><div><small>System</small><h1>Runtime and diagnostics</h1><p>Capabilities are reported by each engine, so unsupported controls are never silently accepted.</p></div></header>
      <section><h2>Engines</h2><div className="engine-grid">{engines.map((engine) => <article className="engine-card" key={`${engine.engine}:${engine.roles.join()}`}><header><div className="model-icon"><Cpu /></div><div><h3>{engine.engine}</h3><p>{engine.roles.join(" · ")} · {engine.version}</p></div><StatusDot healthy={engine.healthy} /></header><div className="capability-list"><span>{engine.streaming ? "Streaming" : "Queued"}</span><span>{engine.tool_calling ? "Tool routing advertised" : "Workflow execution"}</span><span>{engine.settings.length} controls</span>{engine.roles.includes("chat") && <button className="secondary compact-button" onClick={() => toolProbe.mutate()} disabled={toolProbe.isPending}>{toolProbe.isPending ? "Testing…" : "Test structured tools"}</button>}</div></article>)}</div>{toolProbe.data && <div className={`callout ${toolProbe.data.passed ? "success" : "error"}`}>{toolProbe.data.passed ? `Structured tool schema passed on ${toolProbe.data.engine} ${toolProbe.data.version}.` : `Structured tool schema failed: ${toolProbe.data.error || "unknown response"}`}</div>}{toolProbe.error && <div className="callout error">{toolProbe.error.message}</div>}</section>
      <section><h2>Machine</h2>{system.data && <div className="metric-grid"><div><Cpu /><span><strong>{system.data.cpu_count}</strong> logical CPUs</span></div><div><Gauge /><span><strong>{formatBytes(system.data.memory_available_bytes)}</strong> memory free</span></div><div><HardDrive /><span><strong>{formatBytes(system.data.disk_free_bytes)}</strong> disk free</span></div><div><Film /><span><strong>{system.data.ffmpeg_available ? "Ready" : "Missing"}</strong> FFmpeg</span></div></div>}<div className="device-list">{system.data?.devices.map((device) => <div key={device.id}><span className="device-icon"><Cpu size={18} /></span><span><strong>{device.name}</strong><small>{device.backend} · {formatBytes(device.available_memory_bytes)} available</small></span></div>)}</div></section>
      <section>
        <div className="detail-title"><div><h2>Platform support</h2><p>Automated coverage and detected hardware are reported separately from physical certification.</p></div></div>
        {system.data && <div className={`support-summary ${system.data.support.platform_status}`}><div><span className="badge">This machine</span><h3>{system.data.support.platform_label}</h3><p>{system.data.distribution} {system.data.distribution_version} · {system.data.support.accelerator_label}</p></div><div className="support-flags"><span className={`badge ${system.data.support.chat_ready ? "tested" : ""}`}>Chat {system.data.support.chat_ready ? "ready" : "constrained"}</span><span className={`badge ${system.data.support.reference_media_ready ? "tested" : ""}`}>Media {system.data.support.reference_media_ready ? "reference-capable" : "not certified"}</span><span className="badge">{system.data.support.certification_status.replace("-", " ")}</span></div><ul>{system.data.support.messages.map((message) => <li key={message}>{message}</li>)}</ul></div>}
        <div className="platform-grid">{platforms.data?.map((entry) => <article className="platform-card" key={entry.id}><header><div><span className={`badge ${entry.status === "target" ? "likely" : ""}`}>{entry.status}</span><h3>{entry.name}</h3></div><Cpu size={18} /></header><p>{entry.accelerator} · {entry.workloads.join(" · ")}</p>{entry.vram_tiers_gb.length > 0 && <div className="capability-list">{entry.vram_tiers_gb.map((tier) => <span key={tier}>{tier} GB VRAM</span>)}</div>}<small>{entry.evidence}</small></article>)}</div>
      </section>
      <section>
        <div className="detail-title"><div><h2>Model profiles</h2><p>Store load-time and request-time controls independently for every model.</p></div><button className="secondary" onClick={() => profileImport.current?.click()}>Import profile</button></div>
        <input ref={profileImport} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0], "profile"); event.target.value = ""; }} />
        <div className="profile-table interactive">{profiles.data?.map((profile: ModelProfile) => <div key={profile.id}><span className="badge">{profile.role}</span><strong>{profile.name}{profile.is_default ? " · default" : ""}</strong><span>{profile.engine}</span><span className="row-actions">{profile.role === "chat" && profile.model_install_id && <button className="secondary compact-button" onClick={() => loadChat.mutate(profile.id)}>Load</button>}<button className="secondary compact-button" onClick={() => setSelectedProfile(profile)}>Edit</button></span></div>)}</div>
      </section>
      <section>
        <div className="detail-title"><div><h2>Generation presets</h2><p>Reuse sampling and media-generation values across compatible models.</p></div><button className="secondary" onClick={() => presetImport.current?.click()}>Import preset</button></div>
        <input ref={presetImport} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0], "preset"); event.target.value = ""; }} />
        <div className="preset-create"><input aria-label="New preset name" placeholder="New preset name" value={presetName} onChange={(event) => setPresetName(event.target.value)} /><select aria-label="New preset role" value={presetRole} onChange={(event) => setPresetRole(event.target.value as GenerationPreset["role"])}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option></select><button className="primary" disabled={!presetName.trim() || createPreset.isPending} onClick={() => createPreset.mutate()}><Plus size={15} />Create preset</button></div>
        <div className="profile-table interactive">{presets.data?.map((preset) => <div key={preset.id}><span className="badge">{preset.role}</span><strong>{preset.name}{preset.is_default ? " · default" : ""}</strong><span>{Object.keys(preset.settings_json).length} overrides</span><button className="secondary compact-button" onClick={() => setSelectedPreset(preset)}>Edit</button></div>)}</div>
        {(createPreset.error || importError) && <div className="callout error">{createPreset.error?.message || importError}</div>}
      </section>
      <section><h2>Workers</h2><div className="engine-grid">{workers.data?.map((worker) => <article className="engine-card" key={worker.name}><header><div><h3>{worker.name} worker</h3><p>{worker.state === "ready" ? `Ready · PID ${worker.pid}` : worker.state === "starting" ? "Starting and checking health" : worker.state === "exited" ? `Exited · code ${worker.exit_code ?? "unknown"}` : "Stopped or externally managed"}</p></div><StatusDot healthy={worker.state === "ready"} /></header><div className="worker-metrics"><span><strong>{worker.active_jobs}</strong> active</span><span><strong>{worker.queued_jobs}</strong> queued</span><span><strong>{formatBytes(worker.current_memory_bytes)}</strong> current RAM</span><span><strong>{formatBytes(worker.peak_memory_bytes)}</strong> measured peak</span>{worker.estimated_memory_bytes != null && <span><strong>{formatBytes(worker.estimated_memory_bytes)}</strong> estimated load</span>}</div><div className="capability-list">{worker.name === "media" && !worker.running && <button className="secondary compact-button" onClick={() => startMedia.mutate()}>Start ComfyUI</button>}{worker.running && <button className="secondary compact-button" onClick={() => stopWorker.mutate(worker.name)}>Unload</button>}</div></article>)}</div><p className="muted runtime-note">RAM is measured for the managed process tree. Chat load estimates combine GGUF size with a conservative context allowance; actual CPU/GPU placement varies by backend and settings.</p>{(loadChat.error || startMedia.error || stopWorker.error) && <div className="callout error">{(loadChat.error || startMedia.error || stopWorker.error)?.message}</div>}</section>
      <section><div className="detail-title"><div><h2>Database backups</h2><p>Verified SQLite snapshots remain in the local data directory.</p></div><button className="secondary" onClick={() => createBackup.mutate()}>Create backup</button></div><div className="profile-table">{backups.data?.map((backup) => <div key={backup.name}><strong>{backup.name}</strong><span>{formatBytes(backup.size_bytes)}</span><span>{backup.sha256.slice(0, 12)}</span><span>{backup.verified ? "Verified" : "Snapshot"}</span></div>)}</div></section>
      {selectedProfile && <ProfileEditor profile={selectedProfile} engines={engines} onClose={() => setSelectedProfile(null)} />}
      {selectedPreset && <PresetEditor preset={selectedPreset} engines={engines} onClose={() => setSelectedPreset(null)} />}
    </div>
  );
}

function Sidebar({
  projects,
  chats,
  currentChatId,
  view,
  connected,
  onChat,
  onView,
  onNewChat,
  onNewProject,
  onExportProject,
}: {
  projects: Project[];
  chats: Chat[];
  currentChatId: string | null;
  view: View;
  connected: boolean;
  onChat: (id: string) => void;
  onView: (view: View) => void;
  onNewChat: (projectId?: string | null) => void;
  onNewProject: () => void;
  onExportProject: (id: string) => void;
}) {
  const [openProjects, setOpenProjects] = useState<Set<string>>(new Set(projects.map((project) => project.id)));
  const unfiled = chats.filter((chat) => !chat.project_id);
  return (
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Sparkles /></div><span>LM Atelier</span><button className="icon-button mobile-menu"><Menu /></button></div>
      <button className="new-chat" onClick={() => onNewChat(null)}><Plus size={18} />New chat</button>
      <nav className="primary-nav"><button className={view === "models" ? "active" : ""} onClick={() => onView("models")}><Library />Model library</button><button className={view === "workflows" ? "active" : ""} onClick={() => onView("workflows")}><WorkflowIcon />Workflows</button></nav>
      <div className="sidebar-section">
        <div className="section-title"><span>Projects</span><button onClick={onNewProject}><Plus size={15} /></button></div>
        {projects.map((project) => {
          const open = openProjects.has(project.id);
          const projectChats = chats.filter((chat) => chat.project_id === project.id);
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
              </div>
              {open && <div className="chat-list">{projectChats.map((chat) => <button className={view === "chat" && currentChatId === chat.id ? "active" : ""} key={chat.id} onClick={() => onChat(chat.id)}><MessageSquare size={14} /><span>{chat.title}</span></button>)}</div>}
            </div>
          );
        })}
      </div>
      {unfiled.length > 0 && <div className="sidebar-section"><div className="section-title"><span>Chats</span></div><div className="chat-list standalone">{unfiled.map((chat) => <button className={view === "chat" && currentChatId === chat.id ? "active" : ""} key={chat.id} onClick={() => onChat(chat.id)}><MessageSquare size={14} /><span>{chat.title}</span></button>)}</div></div>}
      <div className="sidebar-footer"><button className={view === "settings" ? "active" : ""} onClick={() => onView("settings")}><Settings />Settings</button><div className="connection"><StatusDot healthy={connected} />{connected ? "Local service connected" : "Reconnecting…"}</div></div>
    </aside>
  );
}

function JobsPanel() {
  const client = useQueryClient();
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3_000 });
  const cancel = useMutation({ mutationFn: api.cancelJob, onSuccess: () => void client.invalidateQueries({ queryKey: ["jobs"] }) });
  const active = jobs.data?.filter((job) => ["queued", "running", "paused"].includes(job.status)) ?? [];
  if (!active.length) return null;
  return <div className="jobs-panel"><header><Activity size={16} /><span>{active.length} active job{active.length === 1 ? "" : "s"}</span></header>{active.map((job) => <div className="job-row" key={job.id}><div><strong>{job.kind}</strong><small>{job.phase}</small><div className="progress-track"><div style={{ width: `${Math.max(4, job.progress * 100)}%` }} /></div></div><button className="icon-button" onClick={() => cancel.mutate(job.id)}><CircleStop size={17} /></button></div>)}</div>;
}

export default function App() {
  const client = useQueryClient();
  const [view, setView] = useState<View>("chat");
  const [currentChatId, setCurrentChatId] = useState<string | null>(() => localStorage.getItem("local-lm-chat"));
  const [connected, setConnected] = useState(false);
  const [liveText, setLiveText] = useState<Record<string, string>>({});
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects });
  const chats = useQuery({ queryKey: ["chats"], queryFn: () => api.chats() });
  const chat = useQuery({ queryKey: ["chat", currentChatId], queryFn: () => api.chat(currentChatId!), enabled: Boolean(currentChatId) });
  const engines = useQuery({ queryKey: ["engines"], queryFn: api.engines });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });

  useEffect(() => {
    let dispose: (() => void) | undefined;
    void connectEvents(
      (event: AppEvent) => {
        if (event.type === "text.delta") {
          const messageId = String(event.payload.assistant_message_id ?? "");
          const text = String(event.payload.text ?? "");
          if (messageId) setLiveText((current) => ({ ...current, [messageId]: `${current[messageId] ?? ""}${text}` }));
          return;
        }
        if (event.type.includes("progress") || event.type.startsWith("download.")) void client.invalidateQueries({ queryKey: ["jobs"] });
        if (["run.completed", "run.failed", "run.cancelled"].includes(event.type)) {
          void client.invalidateQueries({ queryKey: ["chat"] });
          void client.invalidateQueries({ queryKey: ["chats"] });
          void client.invalidateQueries({ queryKey: ["jobs"] });
          window.setTimeout(() => setLiveText({}), 200);
        }
      },
      setConnected,
    ).then((cleanup) => { dispose = cleanup; });
    return () => dispose?.();
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
    mutationFn: api.regenerateMessage,
    onSuccess: (accepted) => {
      client.setQueryData<ChatDetail>(["chat", currentChatId], (current) => current ? { ...current, active_head_message_id: accepted.assistant_message.id, messages: [...current.messages, accepted.user_message, accepted.assistant_message] } : current);
      void client.invalidateQueries({ queryKey: ["chats"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const branch = useMutation({
    mutationFn: ({ messageId, text }: { messageId: string; text: string }) => api.branchMessage(messageId, text),
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
  const exportProject = useMutation({
    mutationFn: api.exportProject,
    onSuccess: (artifact) => {
      const link = document.createElement("a");
      link.href = artifact.url;
      link.download = "";
      link.click();
    },
  });

  useEffect(() => {
    if (!currentChatId && chats.data?.length) setCurrentChatId(chats.data[0].id);
  }, [chats.data, currentChatId]);

  const allChats = chats.data ?? [];
  const allProjects = projects.data ?? [];
  const activeContent = useMemo(() => {
    if (view === "models") return <ModelsView />;
    if (view === "workflows") return <WorkflowsView />;
    if (view === "settings") return <SettingsView engines={engines.data ?? []} />;
    return <ChatView chat={chat.data} engines={engines.data ?? []} profiles={profiles.data ?? []} liveText={liveText} onProfile={(field, id) => updateChat.mutate({ [field]: id })} onRegenerate={(messageId) => regenerate.mutate(messageId)} onEdit={(messageId, text) => branch.mutate({ messageId, text })} onStop={() => stop.mutate()} onSend={(text, mode, artifacts, settings) => send.mutate({ text, mode, artifacts, settings })} />;
  }, [view, engines.data, profiles.data, chat.data, liveText, send, regenerate, branch, stop, updateChat]);

  return (
    <div className="app-shell">
      <Sidebar projects={allProjects} chats={allChats} currentChatId={currentChatId} view={view} connected={connected} onChat={(id) => { setCurrentChatId(id); localStorage.setItem("local-lm-chat", id); setView("chat"); }} onView={setView} onNewChat={(projectId) => createChat.mutate(projectId)} onNewProject={() => { const name = window.prompt("Project name"); if (name?.trim()) createProject.mutate(name.trim()); }} onExportProject={(id) => exportProject.mutate(id)} />
      <main>{activeContent}</main>
      <JobsPanel />
      {(send.error || regenerate.error || branch.error || stop.error || createChat.error || createProject.error) && <div className="toast error"><X size={16} />{(send.error || regenerate.error || branch.error || stop.error || createChat.error || createProject.error)?.message}</div>}
    </div>
  );
}
