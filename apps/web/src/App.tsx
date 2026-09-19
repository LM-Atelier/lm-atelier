import { ChatSidebar } from "./ChatSidebar";
import { changeChatPages, restoreChatPages, snapshotChatPages, useChatPages } from "./useChatPages";
import { useAppNavigation } from "./useAppNavigation";
import { ChatWebAccess } from "./ChatWebAccess";
import { ChatSearchConsent } from "./ChatSearchConsent";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Bot,
  LoaderCircle,
  MessageSquare,
  Sparkles,
} from "lucide-react";
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { EditedBranchCards } from "./EditedBranchCards";
import { EmptyState } from "./EmptyState";
import { FirstFailure } from "./FirstFailure";
import { GlobalNotices } from "./GlobalNotices";
import { JobsPanel } from "./JobsPanel";
import { MediaLibraryView } from "./MediaLibraryView";
import { MediaOutputPlan } from "./MediaOutputPlan";
import { ModelsView } from "./ModelsView";
import { PriorTurnEditor } from "./PriorTurnEditor";
import { PromptHelperDialog } from "./PromptHelperDialog";
import { PromptLibraryView } from "./PromptLibraryView";
import { ReferencesLibrary } from "./ReferencesLibrary";
import { SettingsView } from "./SettingsView";
import { SetupSurface } from "./SetupSurface";
import { FirstRunSetup } from "./SetupWizard";
import { StudioView } from "./StudioView";
import { TurnEditor } from "./TurnEditor";
import { WorkspaceComposerDraft } from "./WorkspaceComposerDraft";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { ChatViewProps, PendingTurn } from "./chatComposerContracts";
import {
  EMPTY_COMPOSER_DRAFT,
  updatedComposerDrafts,
  withoutComposerDraft,
  type ComposerDraft, type ComposerPromptSource,
} from "./composerPromptSource";
import type { VisualTarget } from "./libraryEditTargets";
import type { TurnReference } from "./mentionDraft";
import {
  editLineageForResult,
  editSourceUrlForResult,
  priorVisibleMediaByMessage,
} from "./messageMedia";
import { recoverPromptSourceSend } from "./promptSourceSendRecovery";
import { regenerateWithRetry } from "./regenerationRequest";
import { useWorkspaceChrome } from "./sidebarLayout";
import { prefersLessMotion } from "./theme";
import { activeBranchMessages } from "./turnEditorContext";
import type {
  Chat,
  ChatDetail,
  WebSearch,
  EngineRole,
  TurnAccepted,
} from "./types";
import { useAutoSettingsRoles } from "./useAutoSettingsRoles";
import { useEditedBranches } from "./useEditedBranches";
import { useFirstRunSetup } from "./useFirstRunSetup";
import { useLiveEvents } from "./useLiveEvents";
import { useMessageActions } from "./useMessageActions";
import { useProjectMutations } from "./useProjectMutations";
import { useTurnConfirmation } from "./useTurnConfirmation";
import { useWorkPlanMutations } from "./useWorkPlanMutations";
import { focusMainContent, roleForMode } from "./viewHelpers";
import { MessageBubble } from "./MessageBubble";
export { MessageBubble } from "./MessageBubble";

type SendTurnVariables = PendingTurn & {
  chatId: string;
  artifacts: string[];
  settings: Record<string, unknown>;
  /** Subject ids chosen from the mention picker, never parsed from the text. */
  references: TurnReference[];
  outputCount?: number;
  promptSource?: ComposerPromptSource;
  stopCurrent?: boolean;
};

const SETUP_DISMISSED_KEY = "lm-atelier-setup-dismissed";
const CURRENT_CHAT_KEY = "local-lm-chat";

function ChatView({
  onOpenStudio,
  chat,
  engines,
  profiles,
  project,
  liveText,
  pendingTurns,
  workPlans,
  settings,
  settingsRole,
  onSettingsRole,
  presets,
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
  composerDraft,
  onComposerDraft,
}: ChatViewProps) {
  const [editMessageId, setEditMessageId] = useState<string | null>(null);
  const edited = useEditedBranches(chat);
  const endRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const followMessages = useRef(true);
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
  }, [libraryEdit]);
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
  if (!chat) return <EmptyState icon={<MessageSquare />} title="Start a local conversation" body="Create a chat and choose a model. Conversations stay on this machine." />;
  const messages = activeBranchMessages(chat);
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
  const previewMessages = edited.preview && chat.messages.some(
    (message) => message.id === edited.preview?.branch_head_message_id,
  ) ? activeBranchMessages({ ...chat, active_head_message_id: edited.preview.branch_head_message_id }) : [];
  const branchCards = (sourceId?: string) => <EditedBranchCards
    branches={edited.branches.filter((branch) => sourceId
      ? branch.source_message_id === sourceId
      : !messages.some((message) => message.id === branch.source_message_id))}
    activeHeadId={chat.active_head_message_id} activatingPlanId={edited.activatingPlanId}
    onView={edited.view} onContinue={edited.continueBranch}
    onCancelPlan={onCancelPlan} onRetryPlan={onRetryPlan}
    onCancelStep={onCancelStep} onRetryStep={onRetryStep} />;
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
      <div className="chat-heading">
        <div className="chat-header">
          <div><small>{chat.project_id ? "Project chat" : "Unfiled chat"}</small><h1>{chat.title}</h1></div>
        </div>
        <ChatWebAccess chat={chat} />
      </div>
      {/* Reported here because the global list belongs to a component the
          transcript cannot reach. */}
      <FirstFailure of={[feedback, toggleFavorite]} />
      <div className="messages" ref={messagesRef} onScroll={trackMessageScroll}>
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
        {previewMessages.length === 0 && <p>This edited version is being loaded.</p>}
        {previewMessages.map((message) => <MessageBubble key={message.id} message={message}
          liveText={liveText[message.id]} onOpenEdit={setEditMessageId} />)}
      </section>}
      {editMessageId && <PriorTurnEditor key={editMessageId} messageId={editMessageId} chat={chat}
        engines={engines} profiles={profiles} presets={presets}
        maxMediaOutputsPerPlan={maxMediaOutputsPerPlan} PromptHelper={PromptHelperDialog}
        onAccepted={onEditAccepted} onClose={() => setEditMessageId(null)} />}
      <WorkspaceComposerDraft chatId={chat.id} draft={composerDraft} onDraft={onComposerDraft} />
      <TurnEditor PromptHelper={PromptHelperDialog} chat={chat} engines={engines} profiles={profiles} stoppable={stoppable} settings={settings} onSettings={onSettings} settingsRole={settingsRole} onSettingsRole={onSettingsRole} presets={presets} presetId={presetId} onPreset={onPreset} onMode={onMode} onSend={onSend} onStop={onStop} onStopAndSend={onStopAndSend} maxMediaOutputsPerPlan={maxMediaOutputsPerPlan} project={project} visualTarget={visualTarget} quoteTarget={quoteTarget} draft={composerDraft} onDraftChange={onComposerDraft} />
    </div>
  );
}

export default function App() {
  const client = useQueryClient();
  const [turnConfirmDialog, requestTurnConfirmation] = useTurnConfirmation();
  const { view, setView, settingsDestination, setSettingsDestination, settingsFocusRequest, mainFocusRequest } = useAppNavigation();
  useEffect(() => {
    if (mainFocusRequest !== undefined) document.getElementById("main-content")?.focus();
  }, [mainFocusRequest]);
  const { appearance, sidebar } = useWorkspaceChrome();
  const [studioSource, setStudioSource] = useState<{ artifactId: string; chatId: string | null } | null>(null);
  const [modelLibraryRole, setModelLibraryRole] = useState<EngineRole>("chat");
  const [setupOpen, setSetupOpen] = useState<boolean | null>(null);
  const [currentChatId, setCurrentChatId] = useState<string | null>(() => localStorage.getItem(CURRENT_CHAT_KEY));
  const [liveText, setLiveText] = useState<Record<string, string>>({});
  const [chatDrafts, setChatDrafts] = useState<Record<string, Partial<Chat>>>({});
  // Which role's settings the drawer edits while a chat routes in auto: the
  // last role picked in that chat's drawer, chat until one is picked. In every
  // other mode the role follows the mode and this map is ignored.
  const [composerDrafts, setComposerDrafts] = useState<Record<string, ComposerDraft>>({});
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
  const chats = useChatPages();
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
  const applicationInfo = useQuery({ queryKey: ["about"], queryFn: api.about });
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
  const applyAcceptedTurn = useCallback((chatId: string, accepted: TurnAccepted, activate = true) => {
    client.setQueryData<ChatDetail>(["chat", chatId], (current) => {
      if (!current) return current;
      const messageIds = new Set(current.messages.map((message) => message.id));
      const acceptedMessages = [accepted.user_message, accepted.assistant_message]
        .filter((message) => !messageIds.has(message.id));
      return {
        ...current,
        active_head_message_id: activate ? accepted.assistant_message.id : current.active_head_message_id,
        messages: [...current.messages, ...acceptedMessages],
      };
    });
    void client.invalidateQueries({ queryKey: ["chat", chatId], exact: true });
    void client.invalidateQueries({ queryKey: ["chats"] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
    void client.invalidateQueries({ queryKey: ["work-plans", chatId] });
    void client.invalidateQueries({ queryKey: ["edited-branches", chatId] });
  }, [client]);
  const send = useMutation({
    mutationFn: ({ chatId, id, text, mode, artifacts, settings, references, outputCount, promptSource, stopCurrent }: SendTurnVariables) => {
      if (stopCurrent) return api.stopAndSendTurn(
        chatId, text, mode, artifacts, settings, id, references, outputCount,
        promptSource, requestTurnConfirmation,
      );
      return api.sendTurn(
        chatId, text, mode, artifacts, settings, id, "turns", undefined,
        references, outputCount, promptSource, requestTurnConfirmation,
      );
    },
    onMutate: ({ chatId, id, text, mode }) => {
      setPendingTurns((current) => ({
        ...current,
        [chatId]: [...(current[chatId] ?? []), { id, text, mode }],
      }));
    },
    onSuccess: (accepted, { chatId }) => applyAcceptedTurn(chatId, accepted),
    onError: (_error, variables) => recoverPromptSourceSend(client, setComposerDrafts, variables),
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
  const {
    cancelWorkPlan,
    retryWorkPlan,
    cancelWorkStep,
    retryWorkStep,
  } = useWorkPlanMutations(activeChatId);
  const { deleteExchange, removeItem, forkThread } = useMessageActions(setCurrentChatId, setView);
  const regenerate = useMutation({
    mutationFn: ({ chatId, messageId, settings }: { chatId: string; messageId: string; settings: Record<string, unknown> }) =>
      regenerateWithRetry(chatId, messageId, settings),
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
  const stop = useMutation({
    mutationFn: (chatId: string) => api.cancelChat(chatId),
    onSuccess: (_job, chatId) => {
      void client.invalidateQueries({ queryKey: ["chat", chatId] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const updateChat = useMutation({
    mutationFn: ({ id, values }: { id: string; values: Partial<Chat> }) => api.updateChat(id, values),
    onMutate: async ({ id, values }) => {
      await Promise.all([client.cancelQueries({ queryKey: ["chat", id] }), client.cancelQueries({ queryKey: ["chats"] })]);
      const previousChat = client.getQueryData<ChatDetail>(["chat", id]);
      const previousChats = snapshotChatPages(client);
      client.setQueryData<ChatDetail>(["chat", id], (current) => (
        current ? { ...current, ...values } : current
      ));
      changeChatPages(client, (item) => item.id === id ? { ...item, ...values } : item);
      return { previousChat, previousChats };
    },
    onError: (_error, { id }, context) => {
      setChatDrafts((current) => {
        const next = { ...current };
        delete next[id];
        return next;
      });
      if (context?.previousChat) client.setQueryData(["chat", id], context.previousChat);
      if (context?.previousChats) restoreChatPages(client, context.previousChats);
    },
    onSuccess: (updated, { id, values }) => {
      if (updated) {
        client.setQueryData<ChatDetail>(["chat", id], (current) => (
          current ? { ...current, ...updated } : current
        ));
        changeChatPages(client, (item) => item.id === id ? { ...item, ...updated } : item);
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
      const previousChats = snapshotChatPages(client);
      const remainingChats = (chats.data ?? []).filter((candidate) => candidate.id !== deletedId);
      const previousCurrentChatId = currentChatId;
      changeChatPages(client, (item) => item.id === deletedId ? null : item);
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
      setComposerDrafts((current) => withoutComposerDraft(current, deletedId));
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (_error, _deletedChat, context) => {
      if (!context) return;
      restoreChatPages(client, context.previousChats);
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

  const openLibraryImage = useCallback((artifactId: string) => {
    setStudioSource({ artifactId, chatId: null });
    setView("studio");
    focusMainContent();
  }, [setView]);
  const [autoSettingsRoles, rememberSettingsRole] = useAutoSettingsRoles(chats.data, { complete: false });
  const allProjects = useMemo(() => projects.data ?? [], [projects.data]);
  // One place that knows what opening the library means, since three
  // different surfaces send people there.
  const openWorkflows = useCallback(() => { setView("workflows"); focusMainContent(); }, [setView]);
  const activeContent = useMemo(() => {
    if (view === "studio") {
      return (
        <StudioView
          sourceArtifactId={studioSource?.artifactId ?? null}
          sourceChatId={studioSource?.chatId ?? null}
          onOpenArtifact={(artifactId) => setStudioSource({ artifactId, chatId: null })}
          onOpenWorkflows={openWorkflows}
          onClose={() => setStudioSource(null)}/>
      );
    }
    const topLevelView = view === "media" ? <MediaLibraryView onEditImage={openLibraryImage} /> : view === "models" ? <ModelsView key={modelLibraryRole} initialRole={modelLibraryRole} /> : view === "references" ? <ReferencesLibrary /> : view === "prompts" ? <PromptLibraryView /> : view === "workflows" ? <WorkflowsView /> : null;
    if (topLevelView) return topLevelView;
    if (view === "settings") return <SettingsView engines={engines.data ?? []} appearance={appearance} destinationId={settingsDestination} onDestinationChange={setSettingsDestination} focusRequest={settingsFocusRequest} />;
    const displayedChat = chat.data
      ? { ...chat.data, ...(chatDrafts[chat.data.id] ?? {}) }
      : undefined;
    const routingMode = displayedChat?.routing_mode ?? "auto";
    // In auto the settings drawer edits an explicitly chosen role; the backend
    // resolves its role from the operation, so following roleForMode here
    // would silently read and write the chat bag for turns that route to
    // image or video.
    const selectedRole = routingMode === "auto"
      ? (autoSettingsRoles[displayedChat?.id ?? ""] ?? "chat")
      : roleForMode(routingMode);
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
    return <ChatView key={displayedChat?.id ?? "empty-chat"} onOpenStudio={(artifactId) => { setStudioSource({ artifactId, chatId: displayedChat?.id ?? null }); setView("studio"); focusMainContent(); }} chat={displayedChat} engines={engines.data ?? []} profiles={profiles.data ?? []} presets={presets.data ?? []} project={allProjects.find((item) => item.id === displayedChat?.project_id)} liveText={liveText} pendingTurns={displayedChat ? pendingTurns[displayedChat.id] ?? [] : []} workPlans={workPlans.data ?? []} settings={scopedSettings} settingsRole={selectedRole} onSettingsRole={(role) => { if (displayedChat) rememberSettingsRole(displayedChat.id, role); }} presetId={presetId} maxMediaOutputsPerPlan={applicationInfo.data?.max_media_outputs_per_plan ?? 1} composerDraft={displayedChat ? composerDrafts[displayedChat.id] ?? EMPTY_COMPOSER_DRAFT : EMPTY_COMPOSER_DRAFT} onComposerDraft={(update) => {
      if (!displayedChat) return;
      setComposerDrafts((current) => updatedComposerDrafts(current, displayedChat.id, update));
    }} onSettings={(settings) => {
      if (!displayedChat) return;
      persistActiveChat({
        generation_settings_json: {
          ...(displayedChat.generation_settings_json ?? {}),
          [selectedRole]: settings,
        },
      });
    }} onPreset={(selectedPresetId) => {
      if (!displayedChat) return;
      const bindings = { ...(displayedChat.generation_preset_ids_json ?? {}) };
      if (selectedPresetId) bindings[selectedRole] = selectedPresetId;
      else delete bindings[selectedRole];
      persistActiveChat({ generation_preset_ids_json: bindings });
    }} onMode={(mode) => {
      persistActiveChat({ routing_mode: mode });
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
    }} onEditAccepted={(accepted) => {
      if (displayedChat) applyAcceptedTurn(displayedChat.id, accepted, accepted.branch_activated);
    }} onStop={() => {
      if (displayedChat) stop.mutate(displayedChat.id);
    }} onStopAndSend={(text, mode, artifacts, settings, references, outputCount, promptSource) => {
      if (displayedChat) {
        send.mutate({ chatId: displayedChat.id, id: crypto.randomUUID(), text, mode, artifacts, settings, references, outputCount, promptSource, stopCurrent: true });
      }
    }} onDeleteExchange={deleteExchange.mutate} onRemoveItem={removeItem.mutate} onForkThread={forkThread.mutate} onCancelPlan={(planId) => {
      cancelWorkPlan.mutate(planId);
    }} onRetryPlan={(planId) => {
      retryWorkPlan.mutate(planId);
    }} onCancelStep={(stepId) => {
      cancelWorkStep.mutate(stepId);
    }} onRetryStep={(stepId) => {
      retryWorkStep.mutate(stepId);
    }} onSend={(text, mode, artifacts, settings, references, outputCount, promptSource) => {
      if (displayedChat) {
        send.mutate({ chatId: displayedChat.id, id: crypto.randomUUID(), text, mode, artifacts, settings, references, outputCount, promptSource });
      }
    }} />;
  }, [openWorkflows, studioSource, view, setView, appearance, settingsDestination, setSettingsDestination, settingsFocusRequest, modelLibraryRole, engines.data, profiles.data, presets.data, applicationInfo.data, allProjects, chat.data, chatDrafts, autoSettingsRoles, rememberSettingsRole, composerDrafts, liveText, pendingTurns, workPlans.data, send, regenerate, selectResponseRevision, stop, cancelWorkPlan, retryWorkPlan, cancelWorkStep, retryWorkStep, updateChat, deleteExchange, removeItem, forkThread, client, openLibraryImage, applyAcceptedTurn]);

  if (firstRunSetup && setupReadiness.data) {
    return <FirstRunSetup report={setupReadiness.data} onExit={exitFirstRunSetup} onOpenModels={(role) => { exitFirstRunSetup(); setModelLibraryRole(role); setView("models"); }} onOpenWorkflows={() => { exitFirstRunSetup(); setView("workflows"); }} />;
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <ChatSidebar projects={allProjects} engines={engines.data ?? []} presets={presets.data ?? []} currentChatId={activeChatId} view={view} setupState={setupReadiness.data?.state} onSetup={() => setSetupOpen(true)} onChat={(id) => { setCurrentChatId(id); localStorage.setItem(CURRENT_CHAT_KEY, id); setView("chat"); focusMainContent(); }} onView={(nextView) => { setView(nextView); focusMainContent(); }} onNewChat={(projectId) => createChat.mutate(projectId)} onNewProject={(name) => createProject.mutate(name)} onExportProject={(id, includeMedia) => exportProject.mutate({ id, includeMedia })} onImportProject={(file) => importProject.mutate(file)} onUpdateChat={(id, values) => manageChat.mutate({ id, values })} onDeleteChat={(id, deleteGeneratedMedia) => deleteChat.mutate({ id, deleteGeneratedMedia })} onUpdateProject={(id, values) => updateProject.mutate({ id, values })} onDeleteProject={(id) => deleteProject.mutate(id)} sidebar={sidebar} />
      <main id="main-content" tabIndex={-1}>{activeContent}</main>
      <SetupSurface
        open={setupOpen}
        visible={setupVisible}
        report={setupReadiness.data}
        error={setupReadiness.error}
        onRetry={() => void setupReadiness.refetch()}
        onDismiss={() => {
          sessionStorage.setItem(SETUP_DISMISSED_KEY, "1");
          setSetupOpen(false);
        }}
        onClose={() => setSetupOpen(false)}
        onOpenModels={(role) => {
          setModelLibraryRole(role);
          setView("models");
          setSetupOpen(false);
          focusMainContent();
        }}
        onOpenWorkflows={() => { setSetupOpen(false); openWorkflows(); }}
      />
      <JobsPanel />
      {turnConfirmDialog}
      <GlobalNotices connected={eventsConnected} mutations={[send, regenerate, selectResponseRevision, stop, cancelWorkPlan, retryWorkPlan, cancelWorkStep, retryWorkStep, updateChat, createChat, createProject, exportProject, importProject, manageChat, deleteChat, updateProject, deleteProject, deleteExchange, removeItem, forkThread]} />
    </div>
  );
}
