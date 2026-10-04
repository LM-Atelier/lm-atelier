import { useMutation, type QueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { useProjectDeletion } from "./useProjectDeletion";
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
  const deleteProject = useProjectDeletion(client);
  const exportProject = useMutation({
    // Not kept once finished, so a passphrase it was given does not linger.
    gcTime: 0,
    mutationFn: ({ id, includeMedia = true, passphrase }: { id: string; includeMedia?: boolean; passphrase?: string }) =>
      passphrase === undefined ? api.exportProject(id, includeMedia) : api.exportProject(id, includeMedia, passphrase),
    onSuccess: (artifact) => {
      const link = document.createElement("a");
      link.href = artifact.url;
      link.download = "";
      link.click();
    },
  });
  const importProject = useMutation({
    gcTime: 0,
    // A file alone is a plain archive; an encrypted one comes with its passphrase.
    mutationFn: (archive: File | { file: File; passphrase: string }) =>
      archive instanceof File ? api.importProject(archive) : api.importProject(archive.file, archive.passphrase),
    onSuccess: async (project) => {
      void client.invalidateQueries({ queryKey: ["projects"] });
      await client.invalidateQueries({ queryKey: ["chats"] });
      if (!onImportedChat) return;
      const [importedChat] = await api.chatSummaries(project.id, true, "", { limit: 1, offset: 0 });
      if (importedChat) onImportedChat(importedChat.id);
    },
  });
  return { updateProject, deleteProject, exportProject, importProject };
}
