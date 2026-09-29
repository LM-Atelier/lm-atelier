import { MessageSquarePlus } from "lucide-react";

/** Taking the picture on screen to a chat, as a reference for its next message.

It goes to the chat the studio was opened from, or else the chat that was
open, or else a new one, and it is attached rather than sent: the words and
the choice to send stay with the person in the chat. It waits for the
picture's own details, so the chat shows its name and where it came from
rather than an identifier. */
export function StudioUseInChat({ ready, onUse }: { ready: boolean; onUse: () => void }) {
  return (
    <button
      type="button"
      className="secondary compact-button"
      title="Attach this picture to a chat's next message"
      // Not disabled: a focused button that becomes disabled drops focus to
      // the page, and a keyboard user would lose their place.
      aria-disabled={!ready}
      onClick={() => {
        if (ready) onUse();
      }}
    >
      <MessageSquarePlus size={14} aria-hidden="true" /> Use in chat
    </button>
  );
}
