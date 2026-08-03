import { useState, type ReactNode } from "react";
import { AccessibleDialog } from "./AccessibleDialog";

/** Ask before something irreversible, in the app rather than in OS chrome.
 *
 * `window.confirm` cannot say what is about to be lost, cannot show the
 * count, and cannot look any different for "rename this project" than for
 * "run this code on your machine". Everything it guards here is either
 * destructive or a trust decision, so the question is asked where it can
 * carry its own weight.
 */
export function ConfirmDialog({
  title,
  question,
  detail,
  confirmLabel,
  tone = "danger",
  onConfirm,
  onCancel,
}: {
  title: string;
  question: string;
  detail?: ReactNode;
  confirmLabel: string;
  tone?: "danger" | "trust";
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <AccessibleDialog
      title={title}
      eyebrow={tone === "trust" ? "Trust decision" : "Cannot be undone"}
      closeLabel="Close without changing anything"
      onClose={onCancel}
      className="confirm-dialog"
    >
      <p>{question}</p>
      {detail}
      <footer>
        <button className="secondary" onClick={onCancel}>
          Cancel
        </button>
        <button className={tone === "trust" ? "primary" : "secondary danger"} onClick={onConfirm}>
          {confirmLabel}
        </button>
      </footer>
    </AccessibleDialog>
  );
}

/** Ask for a value before acting, in the app rather than in OS chrome. */
export function PromptDialog({
  title,
  label,
  initialValue = "",
  confirmLabel,
  placeholder,
  validate,
  onConfirm,
  onCancel,
}: {
  title: string;
  label: string;
  initialValue?: string;
  confirmLabel: string;
  placeholder?: string;
  validate?: (value: string) => string | null;
  onConfirm: (value: string) => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(initialValue);
  // `window.prompt` accepts anything and reports nothing, so a mistyped
  // commit SHA only failed once the request came back.
  const problem = validate ? validate(value) : null;

  return (
    <AccessibleDialog
      title={title}
      eyebrow="Confirm"
      closeLabel="Close without changing anything"
      onClose={onCancel}
      className="confirm-dialog"
    >
      <label className="prompt-field">
        {label}
        <input
          value={value}
          placeholder={placeholder}
          onChange={(event) => setValue(event.target.value)}
        />
      </label>
      {problem && (
        <p className="prompt-problem" role="alert">
          {problem}
        </p>
      )}
      <footer>
        <button className="secondary" onClick={onCancel}>
          Cancel
        </button>
        <button
          className="primary"
          disabled={Boolean(problem) || !value.trim()}
          onClick={() => onConfirm(value.trim())}
        >
          {confirmLabel}
        </button>
      </footer>
    </AccessibleDialog>
  );
}
