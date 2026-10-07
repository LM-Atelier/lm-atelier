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

  it("lays the tools out in five runs, then undo and redo", () => {
    renderRail();
    const rail = screen.getByRole("navigation", { name: "Editing tools" });
    const runs: string[][] = [[]];
    for (const child of Array.from(rail.children)) {
      if (child.classList.contains("studio-rail-divider")) runs.push([]);
      else runs[runs.length - 1].push(child.getAttribute("aria-label") ?? "");
    }

    expect(runs).toEqual([
      [
        "Crop the picture",
        "Rotate, straighten or flip",
        "Correct the perspective",
        "Resize the picture",
        "Change the canvas size",
        "Adjust light and color",
        "Blur or pixelate part of the picture",
        "Paint over part of the picture",
        "Add text to the picture",
      ],
      ["Select part of the picture"],
      ["Instruct the whole image", "Replace words in the picture", "Remove something from the picture"],
      [
        "Cut the subject out",
        "Replace the background",
        "Replace the subject",
        "Relight from a direction",
        "Extend past the edge",
      ],
      ["Enlarge and restore detail"],
      ["Undo the selection change", "Redo the selection change"],
    ]);
  });

  it("offers the ways of selecting as one tool, marked while any of them is in hand", () => {
    renderRail({ active: "lasso" });

    expect(screen.getByRole("button", { name: "Select part of the picture" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "Instruct the whole image" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    // The panel offers them once Select is chosen; here they are no tools of their own.
    for (const name of [
      "Brush a selection",
      "Erase from the selection",
      "Select a rectangle",
      "Lasso a selection",
      "Fill an area of the selection",
      "Select similar colors",
    ]) {
      expect(screen.queryByRole("button", { name })).toBeNull();
    }
  });

  it("reports the chosen tool, and for Select the way of selecting last used", () => {
    const props = renderRail({ selectionKind: "wand" });
    fireEvent.click(screen.getByRole("button", { name: "Crop the picture" }));
    expect(props.onSelect).toHaveBeenLastCalledWith("crop");
    fireEvent.click(screen.getByRole("button", { name: "Select part of the picture" }));
    expect(props.onSelect).toHaveBeenLastCalledWith("wand");
  });

  it("disables undo and redo until there is something to undo", () => {
    const props = renderRail({ canUndo: false, canRedo: false });
    const undo = screen.getByRole("button", { name: "Undo the selection change" });
    const redo = screen.getByRole("button", { name: "Redo the selection change" });
    expect(undo).toHaveAttribute("aria-disabled", "true");
    expect(redo).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(undo);
    fireEvent.click(redo);
    expect(props.onUndo).not.toHaveBeenCalled();
    expect(props.onRedo).not.toHaveBeenCalled();

    cleanup();
    const live = renderRail({ canUndo: true, canRedo: true });
    fireEvent.click(screen.getByRole("button", { name: "Undo the selection change" }));
    fireEvent.click(screen.getByRole("button", { name: "Redo the selection change" }));
    expect(live.onUndo).toHaveBeenCalledTimes(1);
    expect(live.onRedo).toHaveBeenCalledTimes(1);
    expect(props.onUndo).not.toHaveBeenCalled();
  });

  it("disables everything while no image is loaded", () => {
    const props = renderRail({ disabled: true, canUndo: true, canRedo: true });
    expect(screen.getByRole("button", { name: "Select part of the picture" })).toBeDisabled();
    const undo = screen.getByRole("button", { name: "Undo the selection change" });
    const redo = screen.getByRole("button", { name: "Redo the selection change" });
    expect(undo).toHaveAttribute("aria-disabled", "true");
    expect(redo).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(undo);
    fireEvent.click(redo);
    expect(props.onUndo).not.toHaveBeenCalled();
    expect(props.onRedo).not.toHaveBeenCalled();
  });

  it("guides a tool whose workflow is not installed instead of hiding it", () => {
    // The defect: every selection tool looked ready, and the refusal arrived
    // only after a selection had been drawn and an instruction written.
    render(
      <StudioToolRail
        active="instruct"
        onSelect={vi.fn()}
        onUndo={vi.fn()}
        onRedo={vi.fn()}
        canUndo={false}
        canRedo={false}
        capabilities={[
          { kind: "instruct", workflow_class: "image_to_image", available: true, reason: null, workflow_revision_id: null, adapter_asset_id: null },
          {
            kind: "brush",
            workflow_class: "inpaint",
            available: false,
            reason: "Install an inpainting workflow to edit part of a picture.",
            workflow_revision_id: null,
            adapter_asset_id: null,
          },
        ]}
      />,
    );

    expect(screen.getByRole("button", { name: /Select part of the picture - Install an inpainting/ })).
      toBeInTheDocument();
    // Still clickable: the way out is an install, and a dead button says
    // nothing about that.
    expect(screen.getByRole("button", { name: /Select part of the picture -/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Instruct the whole image" })).toBeInTheDocument();
  });

  it("offers Enhance as a tool of its own", () => {
    render(
      <StudioToolRail
        active="enhance"
        onSelect={vi.fn()}
        onUndo={vi.fn()}
        onRedo={vi.fn()}
        canUndo={false}
        canRedo={false}
        capabilities={[
          { kind: "enhance", workflow_class: "upscale", available: true, reason: null, workflow_revision_id: null, adapter_asset_id: null },
        ]}
      />,
    );

    const enhance = screen.getByRole("button", { name: "Enlarge and restore detail" });
    expect(enhance).toHaveAttribute("aria-pressed", "true");
  });

  it("guides Enhance toward an upscaler when none is installed", () => {
    render(
      <StudioToolRail
        active="instruct"
        onSelect={vi.fn()}
        onUndo={vi.fn()}
        onRedo={vi.fn()}
        canUndo={false}
        canRedo={false}
        capabilities={[
          {
            kind: "enhance",
            workflow_class: "upscale",
            available: false,
            reason: "Install an upscaling workflow to enlarge a picture.",
            workflow_revision_id: null,
            adapter_asset_id: null,
          },
        ]}
      />,
    );

    expect(
      screen.getByRole("button", { name: /Enlarge and restore detail - Install an upscaling/ }),
    ).toBeEnabled();
  });
});
