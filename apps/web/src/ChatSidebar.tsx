import { ChevronDown, Download, Folder, Image as ImageIcon, Library, Menu, MoreHorizontal, Pin, Plus, Quote, Search, Star, Upload, Workflow as WorkflowIcon } from "lucide-react";
import { useRef, useState } from "react";
import { AtelierMark } from "./AtelierMark";
import { ChatManager } from "./ChatManager";
import { PromptDialog } from "./ConfirmDialog";
import { ImageStudioIcon } from "./ImageStudioIcon";
import { ProjectManager } from "./ProjectManager";
import { SidebarFooter } from "./SidebarFooter";
import { SidebarResizer } from "./SidebarResizer";
import type { View } from "./rooms";
import type { SidebarLayout } from "./sidebarLayout";
import type { Chat, Project, EngineCapabilities, GenerationPreset, SetupReadinessReport } from "./types";
import { useChatPages } from "./useChatPages";

export function ChatSidebar({
  projects,
  engines,
  presets,
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
  sidebar,
}: {
  projects: Project[];
  engines: EngineCapabilities[];
  presets: GenerationPreset[];
  currentChatId: string | null;
  view: View;
  setupState?: SetupReadinessReport["state"] | undefined;
  onChat: (id: string) => void;
  onSetup: () => void;
  onView: (view: View) => void;
  onNewChat: (projectId?: string | null) => void;
  onNewProject: (name: string) => void;
  onExportProject: (id: string, includeMedia?: boolean) => void;
  onImportProject: (file: File) => void;
  onUpdateChat: (id: string, values: Partial<Chat>) => void;
  onDeleteChat: (id: string, deleteGeneratedMedia: boolean) => void;
  onUpdateProject: (id: string, values: Partial<Project>) => void;
  onDeleteProject: (id: string) => void;
  sidebar: SidebarLayout;
}) {
  const [naming, setNaming] = useState(false);
  const [closedProjects, setClosedProjects] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [managedChat, setManagedChat] = useState<Chat | null>(null);
  const [managedProject, setManagedProject] = useState<Project | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const projectImport = useRef<HTMLInputElement>(null);
  const chatPages = useChatPages(search, showArchived);
  const chats = chatPages.data ?? [];
  const normalizedSearch = search.trim().toLowerCase();
  const visibleChats = chats.filter((chat) => (showArchived || !chat.archived) && (!normalizedSearch || chat.title.toLowerCase().includes(normalizedSearch)));
  const visibleProjects = projects.filter((project) => (showArchived || !project.archived) && (!normalizedSearch || project.name.toLowerCase().includes(normalizedSearch) || visibleChats.some((chat) => chat.project_id === project.id)));
  const unfiled = visibleChats.filter((chat) => !chat.project_id);
  const chatRow = (chat: Chat) => <div className="sidebar-chat-row" key={chat.id}><button className={`chat-main ${view === "chat" && currentChatId === chat.id ? "active" : ""}`} aria-current={view === "chat" && currentChatId === chat.id ? "page" : undefined} onClick={() => { onChat(chat.id); setMobileOpen(false); }}><span>{chat.title}</span>{chat.archived && <small>Archived</small>}</button><button className={`inline-add sidebar-pin ${chat.pinned ? "pinned" : ""}`} aria-label={chat.pinned ? `Unpin ${chat.title}` : `Pin ${chat.title}`} aria-pressed={chat.pinned} title={chat.pinned ? "Unpin" : "Pin"} onClick={() => onUpdateChat(chat.id, { pinned: !chat.pinned })}><Pin size={13} /></button><button className="inline-add" aria-label={`Manage ${chat.title}`} onClick={() => setManagedChat(chat)}><MoreHorizontal size={13} /></button></div>;
  return (
    <>
    <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`}>
      <div className="brand"><div className="brand-mark"><AtelierMark /></div><span>LM Atelier<small>Local creative studio</small></span><button className="icon-button mobile-menu" aria-label="Toggle navigation" aria-expanded={mobileOpen} onClick={() => setMobileOpen((open) => !open)}><Menu /></button></div>
      <button className="new-chat" onClick={() => { onNewChat(null); setMobileOpen(false); }}><Plus size={18} />New chat</button>
      <nav className="primary-nav"><button className={view === "media" ? "active" : ""} aria-current={view === "media" ? "page" : undefined} onClick={() => { onView("media"); setMobileOpen(false); }}><ImageIcon />Media library</button><button className={view === "models" ? "active" : ""} aria-current={view === "models" ? "page" : undefined} onClick={() => { onView("models"); setMobileOpen(false); }}><Library />Model library</button><button className={view === "references" ? "active" : ""} aria-current={view === "references" ? "page" : undefined} onClick={() => { onView("references"); setMobileOpen(false); }}><Star />References</button><button className={view === "prompts" ? "active" : ""} aria-current={view === "prompts" ? "page" : undefined} onClick={() => { onView("prompts"); setMobileOpen(false); }}><Quote />Prompt library</button><button className={view === "workflows" ? "active" : ""} aria-current={view === "workflows" ? "page" : undefined} onClick={() => { onView("workflows"); setMobileOpen(false); }}><WorkflowIcon />Workflows</button><button className={view === "studio" ? "active" : ""} aria-current={view === "studio" ? "page" : undefined} onClick={() => { onView("studio"); setMobileOpen(false); }}><ImageStudioIcon />Image Studio</button></nav>
      <div className="workspace-search"><Search size={14} /><input aria-label="Search projects and chats" placeholder="Search workspace" maxLength={500} value={search} onChange={(event) => setSearch(event.target.value)} /><button className={showArchived ? "active" : ""} aria-pressed={showArchived} onClick={() => setShowArchived((value) => !value)}>Archived</button></div>
      <div className="workspace-tree" role="region" aria-label="Projects and chats" aria-busy={chatPages.isFetching}>
        <div className="sidebar-section">
          <div className="section-title"><span>Projects</span><input ref={projectImport} hidden type="file" accept=".zip,.lm-atelier.zip,application/zip" onChange={(event) => { const file = event.target.files?.[0]; if (file) onImportProject(file); event.target.value = ""; }} /><button aria-label="Import project" onClick={() => projectImport.current?.click()}><Upload size={14} /></button><button aria-label="New project" onClick={() => setNaming(true)}><Plus size={15} /></button></div>
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
        {chatPages.isPending && <p>Loading chats…</p>}
        {chatPages.isError && <div role="alert">Could not load chats. <button onClick={() => void (chatPages.isFetchNextPageError ? chatPages.fetchNextPage() : chatPages.refetch())}>Try again</button></div>}
        {chatPages.hasNextPage && <button aria-disabled={chatPages.isFetching} onClick={() => { if (!chatPages.isFetching) void chatPages.fetchNextPage(); }}>{chatPages.isFetchingNextPage ? "Loading chats…" : "Load more chats"}</button>}
      </div>
      {naming && <PromptDialog title="New project" label="Project name" confirmLabel="Create project" placeholder="Portrait studies" onCancel={() => setNaming(false)} onConfirm={(name) => { setNaming(false); onNewProject(name); }} />}
      <SidebarFooter setupState={setupState} view={view} onSetup={onSetup} onView={onView} onNavigate={() => setMobileOpen(false)} />
      {managedChat && <ChatManager chat={managedChat} projects={projects} onClose={() => setManagedChat(null)} onSave={(values) => { onUpdateChat(managedChat.id, values); setManagedChat(null); }} onDelete={(deleteGeneratedMedia) => { onDeleteChat(managedChat.id, deleteGeneratedMedia); setManagedChat(null); }} />}
      {managedProject && <ProjectManager project={managedProject} engines={engines} presets={presets} onClose={() => setManagedProject(null)} onSave={(values) => { onUpdateProject(managedProject.id, values); setManagedProject(null); }} onDelete={() => { onDeleteProject(managedProject.id); setManagedProject(null); }} onExport={(includeMedia) => onExportProject(managedProject.id, includeMedia)} />}
    </aside>
      <SidebarResizer layout={sidebar} />
    </>
  );
}
