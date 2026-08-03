import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { StudioToolRail } from "./StudioToolRail";

describe("StudioToolRail", () => {
  afterEach(cleanup);

  function renderRail(overrides: Partial<Parameters<typeof StudioToolRail>[0]> = {}) {
    const props = {
      active: "instruct" as const,
      onSelect: vi.fn(),
      onUndo: vi.fn(),
      onRedo: vi.fn(),
      canUndo: false,
      canRedo: false,
      ...overrides,
    };
    render(<StudioToolRail {...props} />);
    return props;
  }

  it("names every tool and marks the active one", () => {
    renderRail({ active: "brush" });

    expect(screen.getByRole("button", { name: "Brush a selection" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "Instruct the whole image" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    for (const name of [
      "Erase from the selection",
      "Select a rectangle",
      "Lasso a selection",
    ]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument();
    }
  });

  it("reports the chosen tool", () => {
    const props = renderRail();
    fireEvent.click(screen.getByRole("button", { name: "Lasso a selection" }));
    expect(props.onSelect).toHaveBeenCalledWith("lasso");
  });

  it("disables undo and redo until there is something to undo", () => {
    const props = renderRail({ canUndo: false, canRedo: false });
    expect(screen.getByRole("button", { name: "Undo the selection change" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Redo the selection change" })).toBeDisabled();

    cleanup();
    const live = renderRail({ canUndo: true, canRedo: true });
    fireEvent.click(screen.getByRole("button", { name: "Undo the selection change" }));
    fireEvent.click(screen.getByRole("button", { name: "Redo the selection change" }));
    expect(live.onUndo).toHaveBeenCalledTimes(1);
    expect(live.onRedo).toHaveBeenCalledTimes(1);
    expect(props.onUndo).not.toHaveBeenCalled();
  });

  it("disables everything while no image is loaded", () => {
    renderRail({ disabled: true, canUndo: true });
    expect(screen.getByRole("button", { name: "Brush a selection" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Undo the selection change" })).toBeDisabled();
  });
});
