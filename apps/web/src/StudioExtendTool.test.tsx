/** Extend's panel: edges set to reach a shape or an exact size, as well as dragged. */

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioExtendTool } from "./StudioExtendTool";

const NONE = { top: 0, right: 0, bottom: 0, left: 0 };

afterEach(cleanup);

describe("extending to a shape", () => {
  it("sets the edges that reach it, then shows it as the shape the edges make", () => {
    const onMargins = vi.fn();
    const size = { width: 400, height: 300 };
    const { rerender } = render(<StudioExtendTool margins={NONE} size={size} onMargins={onMargins} onReset={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Square" }));

    const squared = { top: 50 / 300, right: 0, bottom: 50 / 300, left: 0 };
    expect(onMargins).toHaveBeenCalledWith(squared);
    rerender(<StudioExtendTool margins={squared} size={size} onMargins={onMargins} onReset={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Square" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "16:9" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("The canvas goes from 400 × 300 to 400 × 400.")).toBeInTheDocument();
  });

  it("holds back the shape the picture is already, and one too far to reach", () => {
    const onMargins = vi.fn();
    const { rerender } = render(
      <StudioExtendTool margins={NONE} size={{ width: 1600, height: 900 }} onMargins={onMargins} onReset={vi.fn()} />,
    );
    const shapes = () => within(screen.getByRole("group", { name: "Extend to a shape" }));

    fireEvent.click(shapes().getByRole("button", { name: "16:9" }));
    expect(shapes().getByRole("button", { name: "16:9" })).toHaveAttribute("aria-disabled", "true");
    expect(shapes().getByRole("button", { name: "Square" })).toHaveAttribute("aria-disabled", "false");

    // 3000 by 500 would need more than twice its height above and below to be square.
    rerender(<StudioExtendTool margins={NONE} size={{ width: 3000, height: 500 }} onMargins={onMargins} onReset={vi.fn()} />);
    fireEvent.click(shapes().getByRole("button", { name: "Square" }));
    expect(shapes().getByRole("button", { name: "Square" })).toHaveAttribute("aria-disabled", "true");
    expect(onMargins).not.toHaveBeenCalled();
  });
});

describe("extending to a size", () => {
  const size = { width: 100, height: 160 };

  it("puts the new room away from where the picture sits", () => {
    const onMargins = vi.fn();
    render(<StudioExtendTool margins={NONE} size={size} onMargins={onMargins} onReset={vi.fn()} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "Width" }), { target: { value: "150" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Height" }), { target: { value: "200" } });
    fireEvent.click(screen.getByRole("button", { name: "Top left" }));
    fireEvent.click(screen.getByRole("button", { name: "Extend to this size" }));

    expect(onMargins).toHaveBeenCalledWith({ top: 0, right: 0.5, bottom: 40 / 160, left: 0 });
  });

  it("refuses a smaller size, and says to crop instead", () => {
    const onMargins = vi.fn();
    render(<StudioExtendTool margins={NONE} size={size} onMargins={onMargins} onReset={vi.fn()} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "Width" }), { target: { value: "90" } });
    fireEvent.click(screen.getByRole("button", { name: "Extend to this size" }));

    expect(screen.getByText("Extend only adds canvas. To make the picture smaller, crop it.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Extend to this size" })).toHaveAttribute("aria-disabled", "true");
    expect(onMargins).not.toHaveBeenCalled();
  });

  it("starts from the size the edges make, whichever way they were set", () => {
    const { rerender } = render(<StudioExtendTool margins={NONE} size={size} onMargins={vi.fn()} onReset={vi.fn()} />);
    expect(screen.getByRole("spinbutton", { name: "Width" })).toHaveValue(100);

    // An edge dragged a quarter of the picture to the right.
    rerender(<StudioExtendTool margins={{ ...NONE, right: 0.25 }} size={size} onMargins={vi.fn()} onReset={vi.fn()} />);
    expect(screen.getByRole("spinbutton", { name: "Width" })).toHaveValue(125);
    expect(screen.getByRole("spinbutton", { name: "Height" })).toHaveValue(160);
  });
});
