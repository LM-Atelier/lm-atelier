import { useState, type ReactNode } from "react";
import { X } from "lucide-react";

export function ErrorCallout({
  message,
  action,
}: {
  message?: string | null;
  action?: ReactNode;
}) {
  const [dismissed, setDismissed] = useState<string | null>(null);
  if (!message || dismissed === message) return null;
  return (
    <div className={`callout error${action ? " action-callout" : ""}`} role="alert">
      <span>{message}</span>
      {action}
      <button
        type="button"
        className="callout-dismiss"
        aria-label="Dismiss error"
        onClick={() => setDismissed(message)}
      >
        <X size={13} />
      </button>
    </div>
  );
}
