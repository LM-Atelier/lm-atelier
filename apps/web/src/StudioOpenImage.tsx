import { useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ImagePlus } from "lucide-react";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { firstImage } from "./studioFiles";

/** Bring an image into the studio without going through a chat first.
 *
 * The studio's only way in used to be an Edit button elsewhere, so opening
 * it directly left a room with no door - and the empty state cheerfully
 * told you to use a button that did not lead here. A picture on disk is the
 * obvious thing to want to edit, and it had no path at all.
 */
export function StudioOpenImage({ onOpened }: { onOpened: (artifactId: string) => void }) {
  const picker = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const open = useMutation({
    mutationFn: (file: File) => api.upload(file),
    onSuccess: (artifact) => onOpened(artifact.id),
  });

  const accept = (files: readonly File[]) => {
    const image = firstImage(files);
    if (image) open.mutate(image);
  };

  return (
    <div
      className={`studio-open-image ${over ? "over" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        accept(Array.from(event.dataTransfer.files));
      }}
    >
      <ImagePlus size={28} aria-hidden="true" />
      <h2>Open an image to edit</h2>
      <p>
        Drop one here, choose a file, or pick Edit on any image in the media library or a chat.
      </p>
      <button
        className="primary"
        disabled={open.isPending}
        onClick={() => picker.current?.click()}
      >
        {open.isPending ? "Opening…" : "Choose an image"}
      </button>
      <input
        ref={picker}
        hidden
        type="file"
        accept="image/*"
        aria-label="Choose an image to edit"
        onChange={(event) => {
          accept(Array.from(event.target.files ?? []));
          event.target.value = "";
        }}
      />
      {open.error && <ErrorCallout message={(open.error as Error).message} />}
    </div>
  );
}
