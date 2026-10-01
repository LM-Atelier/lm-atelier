import { ChatWebAccess } from "./ChatWebAccess";
import { ChatSearchConsent } from "./ChatSearchConsent";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Bot, LoaderCircle, MessageSquare, Sparkles } from "lucide-react";
import { Fragment, useEffect, useLayoutEffect, useRef, useState } from "react";
import { TranscriptReadControls, TranscriptReadFailure } from "./TranscriptReadControls";
import { useChatMessagePages } from "./useChatMessagePages";
import { EditedBranchCards } from "./EditedBranchCards";
import { EmptyState } from "./EmptyState";
import { FirstFailure } from "./FirstFailure";
import { MediaOutputPlan } from "./MediaOutputPlan";
import { PriorTurnEditor } from "./PriorTurnEditor";
import { PromptHelperDialog } from "./PromptHelperDialog";
import { TurnEditor } from "./TurnEditor";
import { WorkspaceComposerDraft } from "./WorkspaceComposerDraft";
import { api } from "./api";
import type { ChatViewProps } from "./chatComposerContracts";
import type { VisualTarget } from "./libraryEditTargets";
import {
  editLineageForResult,
  editSourceUrlForResult,
  priorVisibleMediaByMessage,
} from "./messageMedia";
import { prefersLessMotion } from "./theme";
import { activeBranchMessages } from "./turnEditorContext";
import type { WebSearch } from "./types";
import { useEditedBranches } from "./useEditedBranches";
import { MessageBubble } from "./MessageBubble";
import { useVisibleChatActivity } from "./useVisibleChatActivity";

export function ChatView({
  transcript,
  onOpenStudio,
  chat,
  engines,
  project,
  liveText,
  pendingTurns,
  workPlans,
  settings,
  settingsRole,
  onSettingsRole,
  presetId,
  onSettings,
  onPreset,
  onMode,
  onSend,
  onRegenerate,
  onSelectRevision,
  onEditAccepted,
  onStop,
  onStopAndSend,
  maxMediaOutputsPerPlan,
  onCancelPlan,
  onRetryPlan,
  onCancelStep,
  onRetryStep,
  onDeleteExchange,
  onRemoveItem,
  onForkThread,
  libraryEdit,
  onLibraryEditTaken,
  composerDraft,
  onComposerDraft,
}: ChatViewProps) {
  const [editMessageId, setEditMessageId] = useState<string | null>(null);
  const edited = useEditedBranches(chat);
  const previewPages = useChatMessagePages(edited.preview ? chat?.id : undefined,
    edited.preview?.branch_head_message_id ?? null);
  const endRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  useVisibleChatActivity(messagesRef, chat?.id);
  const followMessages = useRef(true);
  const olderScroll = useRef<{ height: number; top: number } | null>(null);
  useLayoutEffect(() => {
    if (!olderScroll.current || transcript?.loadingOlder) return;
    const viewport = messagesRef.current;
    if (viewport) viewport.scrollTop = olderScroll.current.top + viewport.scrollHeight - olderScroll.current.height;
    olderScroll.current = null;
  }, [chat?.messages, transcript?.loadingOlder]);
  const loadOlder = () => {
    const viewport = messagesRef.current;
    if (!viewport || !transcript) return;
    followMessages.current = false;
    olderScroll.current = { height: viewport.scrollHeight, top: viewport.scrollTop };
    void transcript.loadOlder();
  };
  const previousChatId = useRef<string | undefined>(undefined);
  const [visualTarget, setVisualTarget] = useState<VisualTarget | null>(null);
  const favoriteClient = useQueryClient();
  const feedback = useMutation({
    mutationFn: ({ messageId, revisionId, rating }: {
      messageId: string;
      revisionId: string | null;
      rating: "up" | "down" | null;
    }) => api.setResponseFeedback(messageId, rating, revisionId),
    onSuccess: () => void favoriteClient.invalidateQueries({ queryKey: ["chat"] }),
  });
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
    onLibraryEditTaken?.();
  }, [libraryEdit, onLibraryEditTaken]);
  const [quoteTarget, setQuoteTarget] = useState<{ text: string; requestId: number } | null>(null);
  useEffect(() => {
    if (previousChatId.current !== chat?.id) {
      previousChatId.current = chat?.id;
      followMessages.current = true;
    }
    if (followMessages.current && typeof endRef.current?.scrollIntoView === "function") {
      endRef.current.scrollIntoView({ behavior: prefersLessMotion() ? "auto" : "smooth" });
    }
  }, [chat?.id, chat?.messages, liveText, pendingTurns]);
  const trackMessageScroll = () => {
    const viewport = messagesRef.current;
    if (!viewport) return;
    followMessages.current = (
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight
    ) <= 96;
  };
  if (!chat && (transcript?.loading || transcript?.error)) return <TranscriptReadFailure reads={transcript} />;
  if (!chat) return <EmptyState icon={<MessageSquare />} title="Start a local conversation" body="Create a chat and choose a model. Conversations stay on this machine." />;
  // Paged reads already select the branch; a parent may lie outside this window.
  const messages = transcript ? chat.messages.filter((message) => message.transcript_visible !== false)
    : activeBranchMessages(chat);
  const searchCard = (search: WebSearch) => (
    <ChatSearchConsent key={search.run_id} search={search}
      onChanged={() => {
        void favoriteClient.invalidateQueries({ queryKey: ["chat"] });
        void favoriteClient.invalidateQueries({ queryKey: ["jobs"] });
        void favoriteClient.invalidateQueries({ queryKey: ["work-plans"] });
      }}
      onUseSource={(url) => onComposerDraft((current) => ({
        ...current, text: current.text ? current.text + "\n\n" + url : url,
        promptSource: null,
      }))} />
  );
  const hiddenSearches = (chat.web_searches ?? []).filter((search) =>
    search.job_id !== null && ["awaiting_approval", "approved", "scheduled"].includes(search.state)
    && !messages.some((message) => message.id === search.assistant_message_id));
  const previewMessages = edited.preview
    ? (previewPages.data ?? []).filter((message) => message.transcript_visible !== false) : [];
  const branchCards = (sourceId?: string) => <EditedBranchCards
    branches={edited.branches.filter((branch) => sourceId
      ? branch.source_message_id === sourceId
      : !messages.some((message) => message.id === branch.source_message_id))}
    activeHeadId={chat.active_head_message_id} activatingPlanId={edited.activatingPlanId}
    onView={edited.view} onContinue={edited.continueBranch}
    onCancelPlan={onCancelPlan} onRetryPlan={onRetryPlan}
    onCancelStep={onCancelStep} onRetryStep={onRetryStep} />;
  const priorVisibleMedia = priorVisibleMediaByMessage(messages);
  const stoppable = transcript?.context?.has_pending_response || messages.some(
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
      <div className="chat-heading">
        <div className="chat-header">
          <div><small>{chat.project_id ? "Project chat" : "Unfiled chat"}</small><h1>{chat.title}</h1></div>
        </div>
        <ChatWebAccess chat={chat} />
      </div>
      {/* Reported here because the global list belongs to a component the
          transcript cannot reach. */}
      <FirstFailure of={[feedback, toggleFavorite]} />
      <TranscriptReadFailure reads={transcript?.error ? transcript : undefined} />
      <div className="messages" ref={messagesRef} onScroll={trackMessageScroll}>
        {transcript && <TranscriptReadControls reads={transcript} onOlder={loadOlder} />}
        {hiddenSearches.length > 0 && (
          <section aria-label="Pending searches in other branches">
            <p>Another branch is waiting for your search decision.</p>
            {hiddenSearches.map(searchCard)}
          </section>
        )}
        {messages.length === 0 && pendingTurns.length === 0 ? (
          <EmptyState icon={<Sparkles />} title="What should we make?" body="Ask anything or create an image or video. Auto mode picks the model." />
        ) : messages.map((message, messageIndex) => {
          const messagePlan = planByAssistantMessage.get(message.id);
          const targetPending = message.status === "pending"
            || (message.response_revisions ?? []).some((revision) => revision.status === "pending");
          const compareSourceUrl = message.role === "assistant"
            ? editSourceUrlForResult(messages, messageIndex)
            : null;
          const lineage = compareSourceUrl ? editLineageForResult(messages, messageIndex) : undefined;
          const isPrimaryOutput = messagePlan?.summary_json.assistant_message_id === message.id;
          return (
            <Fragment key={message.id}>
              {messagePlan
                && messagePlan.steps.length > 1
                && isPrimaryOutput
                && (
                  messagePlan.planner_version === "prompt-template-v1"
                  || messagePlan.steps.some((step) => step.status !== "complete")
                ) && (
                <MediaOutputPlan
                  plan={messagePlan}
                  onCancelPlan={onCancelPlan}
                  onRetryPlan={onRetryPlan}
                  onCancelStep={onCancelStep}
                  onRetryStep={onRetryStep}
                />
              )}
              {(chat.web_searches ?? []).filter(
                (search) => search.assistant_message_id === message.id,
              ).map(searchCard)}
              <MessageBubble
                message={message}
                liveText={liveText[message.id]}
                compareSourceUrl={compareSourceUrl}
                lineage={lineage}
                readImageHistory={Boolean(transcript)}
                onFeedback={(messageId, revisionId, rating) =>
                  feedback.mutate({ messageId, revisionId, rating })}
                onToggleFavorite={(part) => part.artifact_id && toggleFavorite.mutate({
                  artifactId: part.artifact_id,
                  next: !part.artifact?.favorite,
                })}
                hiddenInputArtifactIds={priorVisibleMedia.get(message.id)}
                onRegenerate={targetPending ? undefined : (messageId) => onRegenerate(
                  messageId,
                  chat.routing_mode === "auto" ? {} : settings,
                )}
                onSelectRevision={targetPending ? undefined : onSelectRevision}
                onOpenEdit={setEditMessageId}
                onCancelQueued={
                  messagePlan && messagePlan.steps.length <= 1 && messagePlan.status === "queued"
                    ? () => onCancelPlan(messagePlan.id)
                    : undefined
                }
                onOpenStudio={(part) => onOpenStudio(part.artifact_id!)}
                onEditImage={(part, origin) => setVisualTarget({
                  attachment: {
                    id: part.artifact_id!,
                    kind: "image",
                    artifact: part.artifact,
                    origin,
                  },
                  mode: "image",
                  requestId: Date.now(),
                })}
                onAnimateImage={(part, origin) => setVisualTarget({
                  attachment: {
                    id: part.artifact_id!,
                    kind: "image",
                    artifact: part.artifact,
                    origin,
                  },
                  mode: "video",
                  requestId: Date.now(),
                })}
                onReferenceMedia={(part, origin) => setVisualTarget({
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
                onRemoveItem={busy ? undefined : onRemoveItem}
                onForkThread={busy ? undefined : onForkThread}
              />
              {branchCards(message.id)}
            </Fragment>
          );
        })}
        {branchCards()}
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
      {edited.failed && <p role="alert">Edited versions could not be loaded.</p>}
      {edited.activationFailed && <p role="alert">This version could not be selected. Refresh the conversation and try again.</p>}
      {edited.preview && <section role="dialog" aria-label="Edited branch preview">
        <h2>Edited version</h2>
        <button className="secondary" onClick={edited.close}>Close preview</button>
        <button className="secondary" disabled={!edited.preview.can_continue || edited.activating}
          onClick={() => edited.preview && edited.continueBranch(edited.preview)}>Continue from this version</button>
        {previewPages.error && <div role="alert"><p>{previewPages.error.message}</p>
          <button onClick={() => { void previewPages.refetch(); }}>Retry edited version</button></div>}
        {previewPages.isPending && <p>This edited version is being loaded.</p>}
        {previewPages.data && <button aria-disabled={!previewPages.hasNextPage || previewPages.isFetchingNextPage}
          onClick={() => { if (previewPages.hasNextPage && !previewPages.isFetchingNextPage) void previewPages.loadOlder(); }}>
          {previewPages.isFetchingNextPage ? "Loading older preview messages…"
            : previewPages.hasNextPage ? "Load older preview messages" : "All preview messages loaded"}
        </button>}
        {previewMessages.map((message) => <MessageBubble key={message.id} message={message}
          liveText={liveText[message.id]} onOpenEdit={setEditMessageId} readImageHistory={Boolean(transcript)} />)}
      </section>}
      {editMessageId && <PriorTurnEditor key={editMessageId} messageId={editMessageId} chat={chat}
        engines={engines}
        maxMediaOutputsPerPlan={maxMediaOutputsPerPlan} PromptHelper={PromptHelperDialog}
        onAccepted={onEditAccepted} onClose={() => setEditMessageId(null)} />}
      <WorkspaceComposerDraft chatId={chat.id} draft={composerDraft} onDraft={onComposerDraft} />
      <TurnEditor transcriptContext={transcript?.context} PromptHelper={PromptHelperDialog} chat={chat} engines={engines} stoppable={stoppable} settings={settings} onSettings={onSettings} settingsRole={settingsRole} onSettingsRole={onSettingsRole} presetId={presetId} onPreset={onPreset} onMode={onMode} onSend={onSend} onStop={onStop} onStopAndSend={onStopAndSend} maxMediaOutputsPerPlan={maxMediaOutputsPerPlan} project={project} visualTarget={visualTarget} quoteTarget={quoteTarget} draft={composerDraft} onDraftChange={onComposerDraft} />
    </div>
  );
}
