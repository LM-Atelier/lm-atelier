import { Library } from "lucide-react";
import { useId, useState } from "react";
import { LibraryImagePicker } from "./LibraryImagePicker";
import type { StudioSubjectPicture } from "./studioToolState";

/** Where a replaced subject comes from, and what to take from it.
 *
 * The new subject's picture is either chosen from the computer or taken from
 * the library, such as an earlier result; one from the library is sent as the
 * picture it already is, not uploaded again. Choosing either replaces the
 * other. The subject's name is asked for as well, and Replace subject waits
 * for it: the redraw draws what it is told to take, and unnamed it takes the
 * second picture's backdrop too.
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
  const nameHint = useId();
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
          <strong>Name of the new subject</strong>
        </span>
        <input
          type="text"
          value={instruction}
          placeholder="e.g. the dog"
          aria-required="true"
          aria-describedby={nameHint}
          onChange={(event) => onInstructionChange(event.target.value)}
        />
      </label>
      <small id={nameHint}>
        A few words for what to take from the picture. Replace subject waits for them.
      </small>
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
