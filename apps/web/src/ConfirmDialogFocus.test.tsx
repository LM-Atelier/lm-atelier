import { afterEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { ConfirmDialog } from "./ConfirmDialog";

afterEach(cleanup);

it.each(["action", "danger", "trust"] as const)(
  "keeps the %s confirmation focused while pending and allows a later retry",
  (tone) => {
    const confirm = vi.fn();
    const cancel = vi.fn();
    const dialog = (pending: boolean) => (
      <ConfirmDialog title="Apply this change?" question="Review the change before proceeding."
        confirmLabel="Apply change" confirmDisabled={pending} tone={tone}
        onConfirm={confirm} onCancel={cancel} />
    );
    const { rerender } = render(dialog(false));
    const button = screen.getByRole("button", { name: "Apply change" });
    button.focus();
    fireEvent.click(button);
    expect(confirm).toHaveBeenCalledTimes(1);

    rerender(dialog(true));
    expect(button).toHaveAttribute("aria-disabled", "true");
    expect(button).not.toBeDisabled();
    expect(button).toHaveFocus();
    fireEvent.click(button);
    fireEvent.click(button);
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(cancel).not.toHaveBeenCalled();

    rerender(dialog(false));
    expect(button).toHaveAttribute("aria-disabled", "false");
    expect(button).toHaveFocus();
    fireEvent.click(button);
    expect(confirm).toHaveBeenCalledTimes(2);
  },
);

it("refuses an initially unavailable action while preserving cancellation and the focus loop", () => {
  const confirm = vi.fn();
  const cancel = vi.fn();
  render(<ConfirmDialog title="Archive the family?" question="Checking what this would affect."
    confirmLabel="Archive" confirmDisabled onConfirm={confirm} onCancel={cancel} />);
  const button = screen.getByRole("button", { name: "Archive" });
  expect(button).toHaveAttribute("aria-disabled", "true");
  button.focus();
  fireEvent.click(button);
  expect(confirm).not.toHaveBeenCalled();
  fireEvent.keyDown(button, { key: "Tab" });
  const close = screen.getByRole("button", { name: "Close without changing anything" });
  expect(close).toHaveFocus();
  fireEvent.keyDown(close, { key: "Tab", shiftKey: true });
  expect(button).toHaveFocus();
  fireEvent.keyDown(button, { key: "Escape" });
  expect(cancel).toHaveBeenCalledTimes(1);
  expect(confirm).not.toHaveBeenCalled();
});
