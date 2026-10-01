import { Library } from "lucide-react";
import { useState } from "react";
import { LibraryImagePicker } from "./LibraryImagePicker";
import type { StudioSubjectPicture } from "./studioToolState";

/** Where a replaced subject comes from.
 *
 * The new subject's picture is either chosen from the computer or taken from
 * the library, such as an earlier result; one from the library is used as the
 * picture it already is, not uploaded again. Choosing either replaces the
 * other. Its subject is cut out of it and placed where the old one stood, so
 * nothing needs naming: whatever the cutout finds is what is placed.
 */
export function StudioSubjectPanel({
  picture,
  onPicture,
}: {
  picture: StudioSubjectPicture | null;
  onPicture: (picture: StudioSubjectPicture) => void;
}) {
  const [browsing, setBrowsing] = useState(false);
  return (
    <div className="studio-tool-options">
      <small>
        Removes the subject, then places the subject of a second picture where it
        stood, as large as fits its place. Everything around it keeps its own pixels.
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
