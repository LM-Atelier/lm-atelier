import { useQuery } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { ChatManager } from "./ChatManager";
import { api } from "./api";
import type { Chat } from "./types";

export function ChatManagerLoader({ chatId, onClose, onSave, onDelete }: {
  chatId: string;
  onClose: () => void;
  onSave: (values: Partial<Chat>) => void;
  onDelete: (deleteGeneratedMedia: boolean) => void;
}) {
  const detail = useQuery({ queryKey: ["chat-management", chatId], queryFn: ({ signal }) => api.chatMetadata(chatId, signal), staleTime: 0, refetchOnWindowFocus: false, refetchOnReconnect: false });
  if (detail.isSuccess && !detail.isFetching) return <ChatManager key={chatId} chat={detail.data} onClose={onClose} onSave={onSave} onDelete={onDelete} />;
  return <AccessibleDialog title="Chat settings" eyebrow="Conversation" closeLabel="Close chat manager" onClose={onClose} className="workspace-editor">
    {detail.isError ? <p role="alert">Chat settings could not be loaded. <button onClick={() => void detail.refetch()}>Try again</button></p> : <p role="status">Loading chat settings…</p>}
  </AccessibleDialog>;
}
