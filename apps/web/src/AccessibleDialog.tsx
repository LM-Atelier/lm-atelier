import { useEffect, useId, useRef } from "react";
import type { ReactNode } from "react";
import { X } from "lucide-react";

const DIALOG_FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled]):not([type='hidden'])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

function dialogFocusableElements(dialog: HTMLElement): HTMLElement[] {
  return Array.from(dialog.querySelectorAll<HTMLElement>(DIALOG_FOCUSABLE)).filter(
    (element) => element.getAttribute("aria-hidden") !== "true",
  );
}

export function AccessibleDialog({
  title,
  eyebrow,
  closeLabel,
  onClose,
  className = "",
  backdropClassName = "",
  children,
}: {
  title: string;
  eyebrow: string;
  closeLabel: string;
  onClose: () => void;
  className?: string;
  backdropClassName?: string;
  children: ReactNode;
}) {
  const headingId = useId();
  const dialog = useRef<HTMLDivElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);
  const close = useRef(onClose);

  useEffect(() => {
    close.current = onClose;
  }, [onClose]);

  useEffect(() => {
    returnFocus.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const surface = dialog.current;
    const initialFocus = surface?.querySelector<HTMLElement>("[data-dialog-initial-focus]")
      ?? (surface ? dialogFocusableElements(surface)[0] : null)
      ?? surface;
    initialFocus?.focus();
    return () => {
      document.body.style.overflow = previousOverflow;
      if (returnFocus.current?.isConnected) returnFocus.current.focus();
    };
  }, []);

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close.current();
      return;
    }
    if (event.key !== "Tab" || !dialog.current) return;
    const focusable = dialogFocusableElements(dialog.current);
    if (!focusable.length) {
      event.preventDefault();
      dialog.current.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable.at(-1)!;
    if (event.shiftKey && (document.activeElement === first || !dialog.current.contains(document.activeElement))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (document.activeElement === last || !dialog.current.contains(document.activeElement))) {
      event.preventDefault();
      first.focus();
    }
  };

  return (
    <div className={`modal-backdrop ${backdropClassName}`.trim()}>
      {/* A modal owns Escape, and this is the element that holds focus while it
          is open, so the listener belongs here rather than on a control inside. */}
      {/* eslint-disable-next-line jsx-a11y-x/no-noninteractive-element-interactions */}
      <div
        ref={dialog}
        className={`modal ${className}`.trim()}
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId}
        tabIndex={-1}
        onKeyDown={onKeyDown}
      >
        <header>
          <div><small>{eyebrow}</small><h2 id={headingId}>{title}</h2></div>
          <button
            className="icon-button"
            aria-label={closeLabel}
            onClick={onClose}
            data-dialog-initial-focus
          >
            <X />
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}
