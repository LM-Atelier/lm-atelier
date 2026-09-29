/** Correcting the perspective: the panel's answer and its one press. */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioPerspectiveTool } from "./StudioPerspectiveTool";
import { StudioToolOptions } from "./StudioToolOptions";
import { pictureCorners } from "./studioPerspective";
import { initialToolState, studioToolReducer } from "./studioToolState";
import type { StudioPerspective } from "./types";

afterEach(() => {
  cleanup();
});

const SIZE = { width: 120, height: 90 };
const CARD: StudioPerspective = {
  top_left: { x: 30, y: 12 },
  top_right: { x: 96, y: 22 },
  bottom_right: { x: 100, y: 80 },
  bottom_left: { x: 18, y: 70 },
};

function panel(corners: StudioPerspective, busy = false) {
  const onReset = vi.fn();
  const onApply = vi.fn();
  render(<StudioPerspectiveTool corners={corners} size={SIZE} busy={busy} onReset={onReset} onApply={onApply} />);
  return { onReset, onApply };
}

describe("the perspective panel", () => {
  it("offers nothing to reset or apply while the corners are the picture's own", () => {
    const { onReset, onApply } = panel(pictureCorners(120, 90));

    fireEvent.click(screen.getByRole("button", { name: "Reset corners" }));
    fireEvent.click(screen.getByRole("button", { name: "Apply the correction" }));

    expect(screen.getByText("The corners start at the picture's own.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply the correction" })).toHaveAttribute("aria-disabled", "true");
    expect(onReset).not.toHaveBeenCalled();
    expect(onApply).not.toHaveBeenCalled();
  });

  it("says the size it will make, and applies or resets the corners placed", () => {
    const { onReset, onApply } = panel(CARD);

    expect(screen.getByText("Makes a picture 83 by 59 pixels.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Apply the correction" }));
    fireEvent.click(screen.getByRole("button", { name: "Reset corners" }));

    expect(onApply).toHaveBeenCalledWith(CARD);
    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it("holds back corners whose sides cross, and waits while another edit arrives", () => {
    const crossed = { ...CARD, top_left: CARD.top_right, top_right: CARD.top_left };
    const { onApply } = panel(crossed);

    fireEvent.click(screen.getByRole("button", { name: "Apply the correction" }));

    expect(screen.getByText("Keep the corners in their places, with no side crossing another.")).toBeInTheDocument();
    expect(onApply).not.toHaveBeenCalled();
    cleanup();
    const arriving = panel(CARD, true);
    fireEvent.click(screen.getByRole("button", { name: "Apply the correction" }));
    expect(screen.getByText("Applying…")).toBeInTheDocument();
    expect(arriving.onApply).not.toHaveBeenCalled();
  });
});

describe("the perspective tool in the studio's panel", () => {
  it("sends the corners as they stand and puts them back to the picture's own", () => {
    const state = studioToolReducer(
      studioToolReducer({ ...initialToolState(), kind: "perspective" }, { type: "image-changed", width: 120, height: 90 }),
      { type: "set-perspective", corners: CARD },
    );
    const dispatch = vi.fn();
    const onLocalEdit = vi.fn();
    render(
      <StudioToolOptions tools={state} dispatch={dispatch} instruction="" onInstructionChange={vi.fn()} onLocalEdit={onLocalEdit} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Apply the correction" }));
    fireEvent.click(screen.getByRole("button", { name: "Reset corners" }));

    expect(onLocalEdit).toHaveBeenCalledWith("perspective", { perspective: CARD });
    expect(dispatch).toHaveBeenCalledWith({ type: "set-perspective", corners: null });
  });
});
