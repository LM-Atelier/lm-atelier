import { useQuery } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { ChatManager } from "./ChatManager";
import { api } from "./api";
import type { Chat, Project } from "./types";

export function ChatManagerLoader({ chatId, projects, onClose, onSave, onDelete }: {
  chatId: string;
  projects: Project[];
  onClose: () => void;
  onSave: (values: Partial<Chat>) => void;
  onDelete: (deleteGeneratedMedia: boolean) => void;
}) {
  const detail = useQuery({ queryKey: ["chat-management", chatId], queryFn: () => api.chat(chatId), staleTime: 0, refetchOnWindowFocus: false, refetchOnReconnect: false });
  if (detail.isSuccess && !detail.isFetching) return <ChatManager key={chatId} chat={detail.data} projects={projects} onClose={onClose} onSave={onSave} onDelete={onDelete} />;
  return <AccessibleDialog title="Manage chat" eyebrow="Conversation" closeLabel="Close chat manager" onClose={onClose} className="workspace-editor">
    {detail.isError ? <p role="alert">Chat settings could not be loaded. <button onClick={() => void detail.refetch()}>Try again</button></p> : <p role="status">Loading chat settings…</p>}
  </AccessibleDialog>;
}
