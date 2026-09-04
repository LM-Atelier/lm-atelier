/** A trust decision has to be askable in the app, with its facts attached. */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConfirmDialog, PromptDialog } from "./ConfirmDialog";

describe("ConfirmDialog", () => {
  afterEach(() => {
    cleanup();
  });

  it("shows what is at stake and only acts when confirmed", () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(
      <ConfirmDialog
        title="Trust rgthree-comfy?"
        question="Trusting this revision lets its code run inside ComfyUI."
        detail={<code>0f5985d</code>}
        confirmLabel="I reviewed this revision - trust it"
        tone="trust"
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );

    // window.confirm could show none of this: not the revision, not what
    // trusting actually permits, not a label specific to the decision.
    expect(screen.getByText("0f5985d")).toBeInTheDocument();
    expect(screen.getByText(/run inside ComfyUI/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "I reviewed this revision - trust it" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("dismisses on Escape without acting", () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(
      <ConfirmDialog
        title="Remove it?"
        question="This deletes the downloaded repository."
        confirmLabel="Remove"
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );

    // Focus is trapped inside, so Escape is handled where focus lives.
    fireEvent.keyDown(screen.getByRole("button", { name: "Remove" }), { key: "Escape" });
    expect(onCancel).toHaveBeenCalled();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("presents a reversible start as a neutral action", () => {
    render(
      <ConfirmDialog
        title="Start ordered plan?"
        question="This request will run three steps in sequence."
        confirmLabel="Start plan"
        tone="action"
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText("Confirm action")).toBeInTheDocument();
    expect(screen.queryByText("Cannot be undone")).toBeNull();
    expect(screen.getByRole("button", { name: "Start plan" })).toHaveClass("primary");
  });
});

describe("PromptDialog", () => {
  afterEach(() => {
    cleanup();
  });

  it("refuses a value the caller says is wrong", () => {
    const onConfirm = vi.fn();
    render(
      <PromptDialog
        title="Update rgthree-comfy"
        label="Full pinned commit SHA"
        confirmLabel="Pin this revision"
        validate={(value) =>
          /^[0-9a-f]{40}$/i.test(value.trim())
            ? null
            : "A pinned revision is a full 40-character commit SHA."}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    const field = screen.getByLabelText("Full pinned commit SHA");
    fireEvent.change(field, { target: { value: "0f5985d" } });

    // window.prompt accepted anything and said nothing, so a mistyped SHA
    // only failed once the request came back.
    expect(screen.getByRole("alert")).toHaveTextContent("40-character commit SHA");
    expect(screen.getByRole("button", { name: "Pin this revision" })).toBeDisabled();

    fireEvent.change(field, { target: { value: "a".repeat(40) } });
    expect(screen.queryByRole("alert")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Pin this revision" }));
    expect(onConfirm).toHaveBeenCalledWith("a".repeat(40));
  });
});
