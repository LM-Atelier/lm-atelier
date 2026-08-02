import { useMutation, type QueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { Chat, Project } from "./types";

/** The four project mutations, exactly as the workspace root wires them. */
export function useProjectMutations({
  client,
  onImportedChat,
}: {
  client: QueryClient;
  onImportedChat: (chatId: string) => void;
}) {
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
      // Awaited, not timed: a slower refetch used to leave the import on nothing.
      void client.invalidateQueries({ queryKey: ["chats"] }).then(() => {
        const importedChat = client.getQueryData<Chat[]>(["chats"])?.find((item) => item.project_id === project.id);
        if (!importedChat) return;
        onImportedChat(importedChat.id);
      });
    },
  });
  return { updateProject, deleteProject, exportProject, importProject };
}
