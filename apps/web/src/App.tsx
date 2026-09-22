import { ChatSidebar } from "./ChatSidebar";
import { ChatView } from "./ChatView";
import { changeChatPages, restoreChatPages, snapshotChatPages, useChatPages } from "./useChatPages";
import { useChatFieldUpdate } from "./useChatFieldUpdate";
import { useAppNavigation } from "./useAppNavigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTurnSending } from "./useTurnSending";
import { useCallback, useEffect, useMemo, useState } from "react";
import { GlobalNotices } from "./GlobalNotices";
import { JobsPanel } from "./JobsPanel";
import { MediaLibraryView } from "./MediaLibraryView";
import { ModelsView } from "./ModelsView";
import { PromptLibraryView } from "./PromptLibraryView";
import { ReferencesLibrary } from "./ReferencesLibrary";
import { SettingsView } from "./SettingsView";
import { SetupSurface } from "./SetupSurface";
import { FirstRunSetup } from "./SetupWizard";
import { StudioView } from "./StudioView";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { PendingTurn } from "./chatComposerContracts";
import {
  EMPTY_COMPOSER_DRAFT,
  updatedComposerDrafts,
  withoutComposerDraft,
  type ComposerDraft,
} from "./composerPromptSource";
import { regenerateWithRetry } from "./regenerationRequest";
import { useWorkspaceChrome } from "./sidebarLayout";
import type {
  Chat,
  ChatDetail,
  EngineRole,
} from "./types";
import { useAutoSettingsRoles } from "./useAutoSettingsRoles";
import { useFirstRunSetup } from "./useFirstRunSetup";
import { useLiveEvents } from "./useLiveEvents";
import { useMessageActions } from "./useMessageActions";
import { useProjectMutations } from "./useProjectMutations";
import { useTurnConfirmation } from "./useTurnConfirmation";
import { useWorkPlanMutations } from "./useWorkPlanMutations";
import { focusMainContent, roleForMode } from "./viewHelpers";
export { MessageBubble } from "./MessageBubble";

const SETUP_DISMISSED_KEY = "lm-atelier-setup-dismissed";
const CURRENT_CHAT_KEY = "local-lm-chat";

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
  const { applyAcceptedTurn, send } = useTurnSending({
    client, requestTurnConfirmation, setPendingTurns, setComposerDrafts,
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
  const updateChat = useChatFieldUpdate({ client, setChatDrafts });
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
