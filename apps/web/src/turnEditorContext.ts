import type { ChatDetail, Message, RoutingMode, Workflow, WorkflowFamily, WorkflowSelection } from "./types";
import { operationForTurn, revisionForTurn, schemaForRevision } from "./turnWorkflow";

export function activeBranchMessages(chat: ChatDetail): Message[] {
  const visibleMessages = chat.messages.filter(
    (message) => message.transcript_visible !== false,
  );
  if (!chat.active_head_message_id) return visibleMessages;
  const byId = new Map(visibleMessages.map((message) => [message.id, message]));
  const lineage: Message[] = [];
  const visited = new Set<string>();
  let current = byId.get(chat.active_head_message_id);
  while (current && !visited.has(current.id)) {
    visited.add(current.id);
    lineage.unshift(current);
    current = current.parent_id ? byId.get(current.parent_id) : undefined;
  }
  return lineage.length > 0 ? lineage : visibleMessages;
}

export function workflowSchemaForTurn(
  workflows: Workflow[],
  mode: RoutingMode,
  hasAttachments: boolean,
  families: WorkflowFamily[] = [],
  chatSelection: WorkflowSelection | null | undefined = null,
  projectSelection: WorkflowSelection | null | undefined = null,
): Record<string, unknown> | undefined {
  if (mode !== "image" && mode !== "video") return undefined;
  const operation = operationForTurn(mode, hasAttachments);
  const revisionId = revisionForTurn(
    families,
    mode,
    chatSelection,
    projectSelection,
    operation,
  );
  return schemaForRevision(workflows, revisionId, operation);
}
