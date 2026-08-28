import { useCallback, useRef, useState, type ReactNode } from "react";
import { ConfirmDialog } from "./ConfirmDialog";

type Request = {
  title: string;
  question: string;
  detail?: ReactNode;
  confirmLabel: string;
  tone?: "action" | "danger" | "trust";
};

/** Ask a question and wait for the answer, the way window.confirm read.
 *
 * The call sites this replaces were all one-liners, and rewriting each into
 * open-a-dialog plus handle-the-result would have turned a guard into a
 * state machine seven times over. This keeps the shape - ask, branch on the
 * answer - while the question is asked somewhere it can carry its own
 * weight.
 */
export function useConfirm(): [ReactNode, (request: Request) => Promise<boolean>] {
  const [pending, setPending] = useState<Request | null>(null);
  const answer = useRef<((confirmed: boolean) => void) | null>(null);

  const settle = useCallback((confirmed: boolean) => {
    setPending(null);
    answer.current?.(confirmed);
    answer.current = null;
  }, []);

  const confirm = useCallback((request: Request) => {
    // A second question while one is open would strand the first promise
    // unresolved, so the earlier one is answered no before being replaced.
    answer.current?.(false);
    setPending(request);
    return new Promise<boolean>((resolve) => {
      answer.current = resolve;
    });
  }, []);

  const dialog = pending ? (
    <ConfirmDialog
      title={pending.title}
      question={pending.question}
      detail={pending.detail}
      confirmLabel={pending.confirmLabel}
      tone={pending.tone}
      onConfirm={() => settle(true)}
      onCancel={() => settle(false)}
    />
  ) : null;

  return [dialog, confirm];
}
