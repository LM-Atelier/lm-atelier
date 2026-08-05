import { Image as ImageIcon, Paperclip } from "lucide-react";
import { useState } from "react";
import { LibraryAttachPicker } from "./LibraryAttachPicker";
import type { ComposerAttachment } from "./useComposerUploads";

/** The two ways to attach a picture: from disk, or from what is already here.
 *
 * Two buttons rather than one menu. Both are one click, both are used often,
 * and folding them behind a chooser would add a step to the common case to
 * save a few pixels.
 */
export function AttachControls({
  disabled,
  onPickFile,
  onAttach,
}: {
  disabled: boolean;
  onPickFile: () => void;
  onAttach: (attachment: ComposerAttachment) => void;
}) {
  // The picker belongs to the control that opens it. Hoisting its open state
  // into the shell only spreads one interaction across two files.
  const [picking, setPicking] = useState(false);
  return (
    <>
      <button
        className="icon-button"
        onClick={onPickFile}
        disabled={disabled}
        aria-label="Attach file"
      >
        <Paperclip size={18} />
      </button>
      <button
        className="icon-button"
        onClick={() => setPicking(true)}
        disabled={disabled}
        aria-label="Attach from the library"
      >
        <ImageIcon size={18} />
      </button>
      {picking && (
        <LibraryAttachPicker onClose={() => setPicking(false)} onAttach={onAttach} />
      )}
    </>
  );
}
