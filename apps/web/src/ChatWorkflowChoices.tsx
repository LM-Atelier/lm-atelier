import { ActiveChatWorkflowSelector } from "./ActiveChatWorkflowSelector";
import type { RoutingMode } from "./types";
import "./ChatWorkflowChoices.css";

export function ChatWorkflowChoices({ chatId, routingMode }: { chatId: string; routingMode: RoutingMode }) {
  return <div className="chat-workflow-choices" role="group" aria-label="Chat workflow choices">
    <div className="chat-workflow-choice-fields">
      <ActiveChatWorkflowSelector chatId={chatId} routingMode="text" label="Text workflow" />
      <ActiveChatWorkflowSelector chatId={chatId} routingMode="image" label="Image workflow" />
      <ActiveChatWorkflowSelector chatId={chatId} routingMode="video" label="Video workflow" />
    </div>
    {routingMode === "auto" && <small>Auto chooses the request type at send.</small>}
  </div>;
}
