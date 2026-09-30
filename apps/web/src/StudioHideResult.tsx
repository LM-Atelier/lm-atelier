import { EyeOff } from "lucide-react";

/** Take the result on the canvas out of the strip; the picture itself stays in the library. */
export function StudioHideResult({ onHide }: { onHide: () => void }) {
  return (
    <button
      type="button"
      className="secondary compact-button"
      aria-label="Hide this result from the strip"
      title="Takes this result out of the strip. It stays in the library."
      onClick={onHide}
    >
      <EyeOff size={14} aria-hidden="true" /> Hide
    </button>
  );
}
