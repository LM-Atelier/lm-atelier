import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Film, Image as ImageIcon, LoaderCircle, Send } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { editVisionNote, workshopTranscript } from "./promptWorkshop";
import { promptPreviewSettings, resolveCapabilitySettings } from "./settings";
import { currentSendKeyChoice, isSendKeystroke } from "./sendKey";
import { activeBranchMessages } from "./turnEditorContext";
import type { ChatDetail, EngineCapabilities, Message } from "./types";
import { roleForMode } from "./viewHelpers";
import { MessageBubble } from "./MessageBubble";

function promptHelperMessageText(message: Message): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("\n")
    .trim();
}

export function PromptHelperDialog({
  sourceChat,
  initialDraft,
  engines,
  editSourceArtifactIds,
  onAccept,
  onClose,
}: {
  sourceChat: ChatDetail;
  initialDraft: string;
  engines: EngineCapabilities[];
  // When improving an image-edit instruction, the source image rides along so
  // a vision-capable helper grounds its rewrite in what the picture shows.
  editSourceArtifactIds?: string[];
  onAccept: (draft: string) => void;
  onClose: () => void;
}) {
  const client = useQueryClient();
  const started = useRef(false);
  const [adoptedAssistantId, setAdoptedAssistantId] = useState<string | null>(null);
  const [helperId, setHelperId] = useState<string | null>(null);
  const [draft, setDraft] = useState(initialDraft);
  const [instruction, setInstruction] = useState("");
  const [working, setWorking] = useState(true);
  const [closing, setClosing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const helper = useQuery({
    queryKey: ["prompt-helper", helperId],
    queryFn: () => api.promptHelper(helperId!),
    enabled: Boolean(helperId),
    refetchInterval: 500,
  });

  const refresh = useCallback((id: string) => {
    void client.invalidateQueries({ queryKey: ["prompt-helper", id] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
  }, [client]);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    let active = true;
    void (async () => {
      try {
        const created = await api.createPromptHelper(sourceChat.id, initialDraft);
        if (!active) {
          await api.deletePromptHelper(created.id).catch(() => undefined);
          return;
        }
        setHelperId(created.id);
        await api.sendTurn(
          created.id,
          editSourceArtifactIds?.length
            ? "Improve the current draft as an editing instruction for the attached source image. "
              + "Ground it in what the image actually shows and preserve everything it does not ask to change. "
              + "Return the complete revised prompt only."
            : "Improve the current draft. Return the complete revised prompt only.",
          "text",
          editSourceArtifactIds ?? [],
          {},
        );
        if (active) refresh(created.id);
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "Could not start prompt workshop");
      } finally {
        if (active) setWorking(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [editSourceArtifactIds, initialDraft, refresh, sourceChat.id]);

  const helperMessages = helper.data ? activeBranchMessages(helper.data) : [];
  const transcript = workshopTranscript(helperMessages);
  const pending = helperMessages.some((message) => message.status === "pending");
  const latestAssistant = [...helperMessages].reverse().find(
    (message) => message.role === "assistant" && message.status === "complete",
  );
  const latestAssistantText = latestAssistant ? promptHelperMessageText(latestAssistant) : "";
  const visionNote = editVisionNote(latestAssistant, Boolean(editSourceArtifactIds?.length));

  const send = async (mode: "text" | "image" | "video", text: string) => {
    if (!helperId || !draft.trim() || !text.trim()) return;
    setWorking(true);
    setError(null);
    try {
      await api.updatePromptHelper(helperId, draft.trim());
      const role = roleForMode(mode);
      const engine = engines.find((item) => item.roles.includes(role));
      const fields = resolveCapabilitySettings(engine, role);
      await api.sendTurn(
        helperId,
        text.trim(),
        mode,
        [],
        mode === "text" ? {} : promptPreviewSettings(fields),
      );
      setInstruction("");
      refresh(helperId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prompt workshop request failed");
    } finally {
      setWorking(false);
    }
  };

  const finish = async (accept: boolean) => {
    if (accept && !draft.trim()) return;
    setClosing(true);
    setError(null);
    try {
      if (helperId) await api.deletePromptHelper(helperId);
      if (accept) onAccept(draft.trim());
      else onClose();
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not close prompt workshop");
      setClosing(false);
    }
  };

  const unavailable = working || pending || closing || !helperId;
  return (
    <AccessibleDialog
      title="Prompt workshop"
      eyebrow="Draft with a local model"
      closeLabel="Cancel prompt workshop"
      onClose={() => void finish(false)}
      className="prompt-helper-dialog"
    >
      <div className="prompt-helper-body">
        <label className="prompt-helper-draft">
          <span>Draft prompt</span>
          <textarea
            aria-label="Draft prompt"
            value={draft}
            maxLength={20_000}
            rows={7}
            onChange={(event) => setDraft(event.target.value)}
          />
        </label>
        <div className="prompt-helper-conversation" aria-live="polite">
          {transcript.length === 0 && !error && (
            <div className="submission-progress"><LoaderCircle size={17} /><span>Starting workshop…</span></div>
          )}
          {transcript.map((message) => <MessageBubble key={message.id} message={message} />)}
        </div>
        {visionNote && <small className="prompt-helper-vision-note">{visionNote}</small>}
        {latestAssistant && latestAssistantText && adoptedAssistantId !== latestAssistant.id && (
          <button
            type="button"
            className="secondary compact-button prompt-helper-adopt"
            disabled={unavailable}
            onClick={() => {
              setAdoptedAssistantId(latestAssistant.id);
              setDraft(latestAssistantText);
              if (helperId) {
                void api.updatePromptHelper(helperId, latestAssistantText).catch(() => undefined);
              }
            }}
          >
            <Check size={14} /> Use latest response as draft
          </button>
        )}
        <div className="prompt-helper-compose">
          <textarea
            aria-label="Prompt workshop instruction"
            placeholder="Ask for a change…"
            value={instruction}
            rows={2}
            onChange={(event) => setInstruction(event.target.value)}
            onKeyDown={(event) => {
              if (isSendKeystroke(event, currentSendKeyChoice())) {
                event.preventDefault();
                void send("text", instruction);
              }
            }}
          />
          <button
            type="button"
            className="primary compact-button"
            disabled={unavailable || !instruction.trim()}
            onClick={() => void send("text", instruction)}
          >
            <Send size={14} /> Ask helper
          </button>
        </div>
        <div className="prompt-helper-actions">
          <span>
            <button
              type="button"
              className="secondary compact-button"
              disabled={unavailable || !engines.some((engine) => engine.roles.includes("image"))}
              onClick={() => void send("image", draft)}
            >
              <ImageIcon size={14} /> Preview image
            </button>
            <button
              type="button"
              className="secondary compact-button"
              disabled={unavailable || !engines.some((engine) => engine.roles.includes("video"))}
              onClick={() => void send("video", draft)}
            >
              <Film size={14} /> Preview video
            </button>
          </span>
          <span>
            <button type="button" disabled={closing} onClick={() => void finish(false)}>Cancel</button>
            <button
              type="button"
              className="primary"
              disabled={unavailable || !draft.trim()}
              onClick={() => void finish(true)}
            >
              <Check size={15} /> Use prompt
            </button>
          </span>
        </div>
        {(error || helper.error) && (
          <div className="callout error" role="alert">{error || helper.error?.message}</div>
        )}
      </div>
    </AccessibleDialog>
  );
}
