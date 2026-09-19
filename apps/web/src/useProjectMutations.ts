import { useMutation, type QueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { Project } from "./types";

/** The four project mutations, exactly as the workspace root wires them. */
export function useProjectMutations({
  client,
  onImportedChat,
}: {
  client: QueryClient;
  onImportedChat?: (chatId: string) => void;
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
    onSuccess: async (project) => {
      void client.invalidateQueries({ queryKey: ["projects"] });
      await client.invalidateQueries({ queryKey: ["chats"] });
      if (!onImportedChat) return;
      const [importedChat] = await api.chats(project.id, true, "", { limit: 1, offset: 0 });
      if (importedChat) onImportedChat(importedChat.id);
    },
  });
  return { updateProject, deleteProject, exportProject, importProject };
}
