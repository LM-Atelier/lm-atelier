import { EyeOff } from "lucide-react";
import { useSensitiveMediaChoice } from "./sensitiveMedia";

/** A small preview inside a control, covered the same way as a full picture.
 *
 * It sits inside something that is already a button or a link, so it cannot
 * carry a Show button of its own: blurred it is drawn past recognition, and
 * hidden it is replaced by a plain mark and never loaded. Either way it is
 * decoration to assistive technology; the control around it names what it does.
 */
export function ShieldedThumbnail({ src, kind }: { src: string | undefined; kind: "image" | "video" }) {
  const choice = useSensitiveMediaChoice();
  if (choice === "hide") {
    return (
      <span className="shielded-thumbnail-hidden" aria-hidden="true">
        <EyeOff size={16} />
      </span>
    );
  }
  const className = choice === "blur" ? "shielded-thumbnail-blur" : undefined;
  return kind === "video"
    ? <video className={className} src={src} muted preload="metadata" aria-hidden="true" />
    : <img className={className} src={src} alt="" loading="lazy" />;
}
