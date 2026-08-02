import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";

async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Fall back for browsers that refuse the async API outside a secure
  // context; the app runs on plain loopback HTTP.
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("Clipboard access is unavailable");
}

/** Copy with its own confirmation, so the click is not a silent no-op.
 *
 * `buttonText` is empty in the hover action rows, where the icon and its
 * tooltip carry the meaning; the accessible name always states it in full.
 */
export function CopyTextButton({
  text,
  label,
  className,
  buttonText = "Copy",
}: {
  text: string;
  label: string;
  className?: string;
  buttonText?: string;
}) {
  const [copied, setCopied] = useState(false);
  const resetTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => {
    if (resetTimer.current !== undefined) window.clearTimeout(resetTimer.current);
  }, []);
  return (
    <button
      type="button"
      className={className}
      aria-label={copied ? `${label} copied` : label}
      title={copied ? "Copied" : label}
      onClick={() => {
        void copyToClipboard(text).then(() => {
          setCopied(true);
          if (resetTimer.current !== undefined) window.clearTimeout(resetTimer.current);
          resetTimer.current = window.setTimeout(() => setCopied(false), 1_500);
        });
      }}
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {buttonText && <span>{copied ? "Copied" : buttonText}</span>}
    </button>
  );
}
