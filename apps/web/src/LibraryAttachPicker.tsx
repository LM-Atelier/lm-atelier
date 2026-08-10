import { LibraryImagePicker } from "./LibraryImagePicker";
import { artifactOrigin } from "./messageMedia";
import type { ComposerAttachment } from "./useComposerUploads";

/** Attach something already in the library, not only something on disk.
 *
 * The composer could reach a file picker and nothing else, so using a picture
 * the app had itself generated meant exporting it and uploading it back - the
 * same bytes making a round trip through the filesystem to arrive where they
 * started, and arriving as a second artifact with a second identity.
 *
 * Picking from the library attaches the existing artifact by its id, so the
 * reference is the same object the library already holds.
 */
export function LibraryAttachPicker({
  onAttach,
  onClose,
}: {
  onAttach: (attachment: ComposerAttachment) => void;
  onClose: () => void;
}) {
  return (
    <LibraryImagePicker
      title="Attach from the library"
      confirmLabel="Attach"
      onClose={onClose}
      onConfirm={(items) => {
        for (const item of items) {
          onAttach({
            id: item.id,
            kind: "image",
            artifact: item,
            // Whatever it already was. Attaching a generated picture does not
            // make it an upload, and the transcript should not say a person
            // supplied something the app produced.
            origin: artifactOrigin(item) ?? "generated",
          });
        }
      }}
    />
  );
}
