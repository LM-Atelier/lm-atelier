import { useState, type ReactNode } from "react";
import { EyeOff } from "lucide-react";
import { useSensitiveMediaChoice } from "./sensitiveMedia";

/** A picture or video covered until someone chooses to show it.
 *
 * Blurred, it keeps its place and shape but cannot be made out, and it is
 * taken out of the accessibility tree and the keyboard order, so a description
 * or a control inside it cannot give away what the blur covers. Hidden, it is
 * not even loaded. Showing uncovers this one item, and only until it leaves the
 * screen; nothing about it is remembered.
 */
export function ShieldedMedia({ kind, children }: { kind: "image" | "video"; children: ReactNode }) {
  const choice = useSensitiveMediaChoice();
  const [shown, setShown] = useState(false);
  if (choice === "show" || shown) return <>{children}</>;
  const noun = kind === "video" ? "video" : "picture";
  return (
    <div className={`media-shield media-shield-${choice}`}>
      {choice === "blur" && (
        <div className="media-shield-cover" aria-hidden="true" inert>
          {children}
        </div>
      )}
      <div className="media-shield-veil">
        <EyeOff size={18} aria-hidden="true" />
        <span>{choice === "blur" ? `This ${noun} is blurred.` : `This ${noun} is hidden.`}</span>
        <button type="button" className="secondary compact-button" onClick={() => setShown(true)}>
          Show {noun}
        </button>
      </div>
    </div>
  );
}
