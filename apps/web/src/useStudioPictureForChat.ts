import { useCallback, useState } from "react";
import type { VisualTarget } from "./libraryEditTargets";
import type { MediaOrigin } from "./messageMedia";
import type { Artifact } from "./types";

/** A picture from Image Studio, as a chat's composer attaches it. */
export interface StudioPictureForChat {
  artifactId: string;
  artifact: Artifact | null;
  origin: MediaOrigin;
}

/** Handing a picture from Image Studio to a chat's composer, once.

The picture goes to the chat named, or to a new chat when none is. It arrives
as a reference, the way the Reference action in a chat attaches one, so the
chat's mode is left as it was. It is handed down while that chat is showing
until the chat says it has taken it, and let go then, so opening the chat
again later does not attach the picture a second time. */
export function useStudioPictureForChat({
  displayedChatId,
  openChat,
  startChat,
}: {
  /** The chat whose view is showing, if one is. */
  displayedChatId: string | null;
  /** Show a chat that already exists. */
  openChat: (chatId: string) => void;
  /** Start a new chat, and report its id once it exists. */
  startChat: (onCreated: (chatId: string) => void) => void;
}) {
  const [pending, setPending] = useState<{ chatId: string; target: VisualTarget } | null>(null);
  const handOver = pending && pending.chatId === displayedChatId ? pending.target : null;

  const sendToChat = useCallback((picture: StudioPictureForChat, chatId: string | null) => {
    const target: VisualTarget = {
      attachment: { id: picture.artifactId, kind: "image", artifact: picture.artifact, origin: picture.origin },
      mode: null,
      requestId: Date.now(),
    };
    if (chatId) {
      setPending({ chatId, target });
      openChat(chatId);
    } else {
      startChat((created) => setPending({ chatId: created, target }));
    }
  }, [openChat, startChat]);

  /** The chat has attached the picture; stop handing it down. */
  const taken = useCallback(() => setPending(null), []);

  return { handOver, sendToChat, taken };
}
