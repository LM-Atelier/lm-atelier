import { X } from "lucide-react";

/** Put the picture down and go back to an empty studio.
 *
 * Closing loses nothing, so it asks nothing. Every result is in the media
 * library, and the picture's session, every edit included, comes back
 * when the same picture is opened in the Studio again. An edit still running
 * carries on after the Studio closes and is there on return. The one thing
 * closing would cut short is a replacement between its two edits, since the
 * second is sent from here once the first is done; until then the button waits.
 */
export function StudioCloseButton({ halfway, onClose }: { halfway: boolean; onClose: () => void }) {
  return (
    <button
      type="button"
      className="secondary compact-button"
      // Not disabled: a focused button that becomes disabled drops focus to the page.
      aria-disabled={halfway}
      title={halfway ? "Closing waits until the replacement's second edit is sent" : undefined}
      onClick={() => !halfway && onClose()}
    >
      <X size={14} aria-hidden="true" /> Close
    </button>
  );
}
