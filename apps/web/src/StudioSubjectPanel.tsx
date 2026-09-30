import { Library } from "lucide-react";
import { useState } from "react";
import { LibraryImagePicker } from "./LibraryImagePicker";
import type { StudioSubjectPicture } from "./studioToolState";

/** Where a replaced subject comes from, and what to take from it.
 *
 * The new subject's picture is either chosen from the computer or taken from
 * the library, such as an earlier result; one from the library is sent as the
 * picture it already is, not uploaded again. Choosing either replaces the
 * other.
 */
export function StudioSubjectPanel({
  picture,
  instruction,
  onPicture,
  onInstructionChange,
}: {
  picture: StudioSubjectPicture | null;
  instruction: string;
  onPicture: (picture: StudioSubjectPicture) => void;
  onInstructionChange: (instruction: string) => void;
}) {
  const [browsing, setBrowsing] = useState(false);
  return (
    <div className="studio-tool-options">
      <small>
        Cuts the subject out first, then redraws it from a second picture. Everything
        around it keeps its own pixels.
      </small>
      <label>
        <span>
          <strong>Picture of the new subject</strong>
        </span>
        <input
          type="file"
          accept="image/*"
          onChange={(event) => {
            // A dialog closed without a choice keeps the picture already chosen.
            const chosen = event.target.files?.[0];
            if (chosen) onPicture(chosen);
          }}
        />
      </label>
      <button type="button" className="secondary compact-button" onClick={() => setBrowsing(true)}>
        <Library size={14} aria-hidden="true" /> Choose from the library
      </button>
      {picture && <small>{picture.name}</small>}
      <label>
        <span>
          <strong>What to take from it</strong> (optional)
        </span>
        <input
          type="text"
          value={instruction}
          placeholder="e.g. the dog"
          onChange={(event) => onInstructionChange(event.target.value)}
        />
      </label>
      {browsing && (
        <LibraryImagePicker
          title="Picture of the new subject"
          confirmLabel="Use this picture"
          single
          onConfirm={([item]) => {
            if (item) onPicture({ artifactId: item.id, name: item.original_name ?? "A picture from the library" });
          }}
          onClose={() => setBrowsing(false)}
        />
      )}
    </div>
  );
}
