import type { RecoveryCommand, RecoveryImpact, RecoveryItem, RecoveryKind, RecoveryPage, RecoveryResult, RecoveryState } from "./recoveryTypes";
import type { RecoveryBatchCommand, RecoveryBatchPreview, RecoveryBatchResult, RecoveryBatchSelection } from "./recoveryTypes";

type Request = <T>(path: string, init?: RequestInit) => Promise<T>;

export function recoveryApi(request: Request) {
  const itemPath = (id: string) => `/api/recovery-items/${encodeURIComponent(id)}`;
  const post = <T>(path: string, payload: object) => request<T>(path, { method: "POST", body: JSON.stringify(payload) });
  return {
    previewRecoveryBatch: (selection: RecoveryBatchSelection, signal?: AbortSignal) =>
      request<RecoveryBatchPreview>("/api/recovery-items/batches", { method: "POST", body: JSON.stringify(selection), signal }),
    applyRecoveryBatch: (batchId: string, command: RecoveryBatchCommand) =>
      post<RecoveryBatchResult>(`/api/recovery-items/batches/${encodeURIComponent(batchId)}/apply`, command),
    recoveryItems: (options: { cursor?: string; state?: RecoveryState; kind?: RecoveryKind; deletedSince?: string; signal?: AbortSignal } = {}) => {
      const parameters = new URLSearchParams();
      if (options.cursor) parameters.set("cursor", options.cursor);
      if (options.state) parameters.set("state", options.state);
      if (options.kind) parameters.set("kind", options.kind);
      if (options.deletedSince) parameters.set("deleted_since", options.deletedSince);
      return request<RecoveryPage>(`/api/recovery-items?${parameters}`, { signal: options.signal });
    },
    recoveryImpact: (id: string, signal?: AbortSignal) => request<RecoveryImpact>(`${itemPath(id)}/impact`, { signal }),
    restoreRecovery: (id: string, command: RecoveryCommand & { restore_unfiled: boolean }) =>
      post<RecoveryResult>(`${itemPath(id)}/restore`, command),
    purgeRecovery: (id: string, command: RecoveryCommand & { acknowledgement: "permanently-delete" }) =>
      post<RecoveryResult>(`${itemPath(id)}/purge`, command),
    deletionImpact: (chatId: string, deleteGeneratedMedia = false, signal?: AbortSignal) =>
      request<RecoveryImpact>(`/api/chats/${encodeURIComponent(chatId)}/deletion-impact?delete_generated_media=${deleteGeneratedMedia}`, { signal }),
    trashChat: (chatId: string, command: RecoveryCommand & { delete_generated_media: boolean }) =>
      post<RecoveryItem>(`/api/chats/${encodeURIComponent(chatId)}/trash`, command),
    projectDeletionImpact: (projectId: string, signal?: AbortSignal) =>
      request<RecoveryImpact>(`/api/projects/${encodeURIComponent(projectId)}/deletion-impact`, { signal }),
    trashProject: (projectId: string, command: RecoveryCommand) =>
      post<RecoveryItem>(`/api/projects/${encodeURIComponent(projectId)}/trash`, command),
    workflowDeletionImpact: (familyId: string, signal?: AbortSignal) =>
      request<RecoveryImpact>(`/api/workflow-families/${encodeURIComponent(familyId)}/deletion-impact`, { signal }),
    trashWorkflow: (familyId: string, command: RecoveryCommand) =>
      post<RecoveryItem>(`/api/workflow-families/${encodeURIComponent(familyId)}/trash`, command),
    mediaDeletionImpact: (entryId: string, signal?: AbortSignal) =>
      request<RecoveryImpact>(`/api/artifact-library/${encodeURIComponent(entryId)}/deletion-impact`, { signal }),
    trashMedia: (entryId: string, command: RecoveryCommand) =>
      post<RecoveryItem>(`/api/artifact-library/${encodeURIComponent(entryId)}/trash`, command),
  };
}
