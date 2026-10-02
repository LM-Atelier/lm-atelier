import { ActiveChatWorkflowSelector } from "./ActiveChatWorkflowSelector";
import { useId, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { RoutingMode } from "./types";
import "./ChatWorkflowChoices.css";

export function ChatWorkflowChoices({ chatId }: { chatId: string; routingMode: RoutingMode }) {
  return <ChatWorkflowPanel key={chatId} chatId={chatId} />;
}

function ChatWorkflowPanel({ chatId }: { chatId: string }) {
  const id = useId();
  const storageKey = `local-lm-workflow-controls:${chatId}`;
  const [visible, setVisible] = useState(() => {
    try { return localStorage.getItem(storageKey) !== "hidden"; } catch { return true; }
  });
  return <div className="chat-workflow-choices" role="group" aria-label="Chat workflow choices">
    <button type="button" className="secondary compact-button chat-workflow-toggle" aria-expanded={visible}
      aria-controls={id} onClick={() => {
        const next = !visible;
        setVisible(next);
        try { localStorage.setItem(storageKey, next ? "shown" : "hidden"); } catch { /* The control remains usable without persistence. */ }
      }}>
      {visible ? <ChevronUp size={14} aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}
      {visible ? "Hide workflows" : "Show workflows"}
    </button>
    {visible && <div id={id} className="chat-workflow-choice-fields">
      <ActiveChatWorkflowSelector chatId={chatId} routingMode="text" label="Text workflow" />
      <ActiveChatWorkflowSelector chatId={chatId} routingMode="image" label="Image workflow" />
      <ActiveChatWorkflowSelector chatId={chatId} routingMode="video" label="Video workflow" />
    </div>}
  </div>;
}
