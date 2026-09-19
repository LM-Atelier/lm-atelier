import {
  Bot,
  ChevronLeft,
  ChevronRight,
  GitBranch,
  Quote,
  RotateCcw,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react";
import { Fragment, useState } from "react";
import { ArtifactPart } from "./ArtifactPart";
import { CopyTextButton } from "./CopyTextButton";
import { MarkdownText } from "./MarkdownText";
import { MentionText } from "./MentionText";
import { MessageRemovalConfirmation, UserMessageControls } from "./MessageRemovalControls";
import { MessageTimestamp } from "./MessageTimestamp";
import { PendingResponseStatus } from "./PendingResponseStatus";
import { editReviewSummary } from "./editReview";
import { editStrengthNote } from "./editStrengthNote";
import {
  mediaOriginForPart,
  messagePartsForTranscript,
  type EditLineageStep,
  type MediaOrigin,
} from "./messageMedia";
import type {
  Message,
  MessagePart,
  MessageReference,
} from "./types";
import { videoLengthSummary } from "./videoLength";

/** The library's Edit action: attach the selection in the chat composer,
 * switch to image mode, and open the studio. */
function PartView({
  part,
  liveText,
  markdown = false,
  references,
  origin,
  onEditImage,
  onOpenStudio,
  onAnimateImage,
  onReferenceMedia,
  onToggleFavorite,
  compareSourceUrl,
  lineage,
}: {
  part: MessagePart;
  liveText?: string;
  markdown?: boolean;
  /** What the turn recorded referring to, so a text part can mark exactly
   *  those and nothing it found by reading the prose. */
  references?: MessageReference[];
  origin: MediaOrigin | null;
  onEditImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onOpenStudio?: (part: MessagePart) => void;
  onAnimateImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onReferenceMedia?: (part: MessagePart, origin: MediaOrigin) => void;
  onToggleFavorite?: (part: MessagePart) => void;
  compareSourceUrl?: string | null;
  lineage?: EditLineageStep[];
}) {
  if (part.type === "text") {
    const text = liveText || part.text || "";
    return markdown ? <MarkdownText text={text} /> : <MentionText text={text} references={references} />;
  }
  if (part.type === "image" || part.type === "video" || part.type === "attachment") {
    return <ArtifactPart part={part} origin={origin} onEditImage={onEditImage} onOpenStudio={onOpenStudio} onAnimateImage={onAnimateImage} onReferenceMedia={onReferenceMedia} onToggleFavorite={onToggleFavorite} compareSourceUrl={compareSourceUrl} lineage={lineage} />;
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

export function MessageBubble({
  message,
  liveText,
  hiddenInputArtifactIds,
  onRegenerate,
  onEdit,
  onOpenEdit,
  onSelectRevision,
  onCancelQueued,
  onEditImage,
  onOpenStudio,
  onAnimateImage,
  onReferenceMedia,
  onToggleFavorite,
  onQuote,
  onDeleteExchange,
  onRemoveItem,
  onForkThread,
  onFeedback,
  compareSourceUrl,
  lineage,
}: {
  message: Message;
  liveText?: string;
  hiddenInputArtifactIds?: ReadonlySet<string>;
  onRegenerate?: (messageId: string) => void;
  onEdit?: (messageId: string, text: string) => void;
  onOpenEdit?: (messageId: string) => void;
  onSelectRevision?: (messageId: string, revisionId: string) => void;
  onCancelQueued?: () => void;
  onEditImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onOpenStudio?: (part: MessagePart) => void;
  onAnimateImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onReferenceMedia?: (part: MessagePart, origin: MediaOrigin) => void;
  onToggleFavorite?: (part: MessagePart) => void;
  onQuote?: (text: string) => void;
  onDeleteExchange?: (messageId: string) => void;
  onRemoveItem?: (messageId: string) => void;
  onForkThread?: (messageId: string) => void;
  onFeedback?: (messageId: string, revisionId: string | null, rating: "up" | "down" | null) => void;
  compareSourceUrl?: string | null;
  lineage?: EditLineageStep[];
}) {
  const contentRemoved = Boolean(message.content_removed_at);
  const visibleParts = contentRemoved
    ? []
    : messagePartsForTranscript(message, hiddenInputArtifactIds);
  const userText = visibleParts.filter((part) => part.type === "text").map((part) => part.text || "").join("\n");
  const copyableText = (contentRemoved ? "" : liveText || userText).trim();
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
  const [confirmingRemoval, setConfirmingRemoval] = useState(false);
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
  const removalConfirmation = confirmingRemoval ? (
    <MessageRemovalConfirmation messageId={message.id} onRemove={(id) => { setConfirmingRemoval(false); onRemoveItem?.(id); }} onKeep={() => setConfirmingRemoval(false)} />
  ) : null;
  const userMessageMeta = !contentRemoved && message.role === "user" && message.status === "complete" && !editing ? (
    <UserMessageControls
      messageId={message.id}
      createdAt={message.created_at}
      copyableText={copyableText}
      onEdit={onOpenEdit ? () => onOpenEdit(message.id) : onEdit ? () => setEditing(true) : undefined}
      onRemoveItem={onRemoveItem}
      onDeleteExchange={onDeleteExchange}
    />
  ) : null;
  const messageActionPartIndex = userMessageMeta ? renderedParts.map((part) => part.type).lastIndexOf("text") : -1;
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "You" : <Bot size={19} />}</div>
      <div className="message-content">
        {contentRemoved ? <div className="message-removed">Message removed</div> : editing ? <div className="message-edit"><textarea aria-label="Edit message" rows={4} value={draft} onChange={(event) => setDraft(event.target.value)} /><div><button onClick={() => { setDraft(userText); setEditing(false); }}>Cancel</button><button className="primary" disabled={!draft.trim()} onClick={() => { onEdit?.(message.id, draft.trim()); setEditing(false); }}>Send edited message</button></div></div> : renderedParts.map((part, index) => <Fragment key={part.id}><PartView part={part} liveText={liveText} markdown={message.role === "assistant"} references={message.references} origin={mediaOriginForPart(part, operation, message.role === "assistant" ? "generated" : null)} onEditImage={onEditImage} onOpenStudio={onOpenStudio} onAnimateImage={onAnimateImage} onReferenceMedia={onReferenceMedia} onToggleFavorite={onToggleFavorite} compareSourceUrl={message.role === "assistant" ? compareSourceUrl : undefined} lineage={message.role === "assistant" ? lineage : undefined} />{index === messageActionPartIndex && userMessageMeta}</Fragment>)}
        {!contentRemoved && liveText && !visibleParts.some((part) => part.type === "text") && (
          <MarkdownText text={liveText} />
        )}
        {!contentRemoved && showChatStartup && <PendingResponseStatus label={chatProgress?.text || "Starting chat"} startedAt={message.created_at} />}
        {messageActionPartIndex < 0 && userMessageMeta}
        {message.role === "assistant" && message.status === "cancelled" && !visibleParts.some((part) => part.type === "error") && (
          <div className="message-meta"><span>Generation cancelled</span></div>
        )}
        {!contentRemoved && message.role === "assistant" && message.status === "complete" && (
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
            {editReviewSummary(provenance) && <span>{editReviewSummary(provenance)}</span>}
            {editStrengthNote(provenance) && <span>{editStrengthNote(provenance)}</span>}
            {videoLengthSummary(provenance) && <span>{videoLengthSummary(provenance)}</span>}
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
            {removalConfirmation ?? <span className="message-actions">
              {onFeedback && (() => {
                const displayed = completedRevisions[revisionIndex] ?? null;
                const current = displayed ? displayed.feedback ?? null : message.feedback ?? null;
                const send = (rating: "up" | "down") => onFeedback(
                  message.id,
                  displayed?.id ?? null,
                  current === rating ? null : rating,
                );
                return (
                  <>
                    <button
                      aria-label="Good response"
                      title="Good response (stored locally)"
                      aria-pressed={current === "up"}
                      onClick={() => send("up")}
                    >
                      <ThumbsUp size={14} fill={current === "up" ? "currentColor" : "none"} />
                    </button>
                    <button
                      aria-label="Poor response"
                      title="Poor response (stored locally)"
                      aria-pressed={current === "down"}
                      onClick={() => send("down")}
                    >
                      <ThumbsDown size={14} fill={current === "down" ? "currentColor" : "none"} />
                    </button>
                  </>
                );
              })()}
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
              {onRemoveItem && (
                <button aria-label="Remove this item, keep replies" title="Remove this item, keep replies" onClick={() => setConfirmingRemoval(true)}>
                  <X size={14} />
                </button>
              )}
            </span>}
          </div>
        )}
        {!contentRemoved && message.role === "assistant" && message.status !== "complete" && copyableText && (
          <div className="message-meta"><CopyTextButton text={copyableText} label="Copy assistant message" /></div>
        )}
        {!contentRemoved && message.role === "assistant" && message.status === "pending" && onCancelQueued && (
          <div className="message-meta">
            <button onClick={onCancelQueued}><X size={13} /> Cancel queued item</button>
          </div>
        )}
      </div>
    </article>
  );
}
