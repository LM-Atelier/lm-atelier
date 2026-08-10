import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { ArtifactPart } from "./ArtifactPart";
import type { MessagePart } from "./types";

/** The media actions are icons now, but each stays a distinct, named
 * operation - Edit selects image mode, Animate seeds a video turn,
 * Reference attaches without changing the mode. Collapsing them into one
 * ambiguous "attach" would be a product regression, not a visual one. */

function imagePart(): MessagePart {
  return {
    id: "part-1",
    position: 0,
    type: "image",
    text: null,
    artifact_id: "sha256:image",
    metadata_json: {},
  };
}

describe("media action row", () => {
  afterEach(cleanup);

  it("exposes every action as a labeled control without visible label text", () => {
    render(
      <ArtifactPart
        part={imagePart()}
        origin="generated"
        onEditImage={vi.fn()}
        onOpenStudio={vi.fn()}
        onAnimateImage={vi.fn()}
        onReferenceMedia={vi.fn()}
        onToggleFavorite={vi.fn()}
      />,
    );

    for (const name of [
      "Edit this image",
      "Open this image in the Image Studio",
      "Animate this image",
      "Reference this media",
      "Favorite this image",
    ]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument();
    }
    expect(screen.getByRole("link", { name: "Download this image" })).toBeInTheDocument();
    const edit = screen.getByRole("button", { name: "Edit this image" });
    const studio = screen.getByRole("button", {
      name: "Open this image in the Image Studio",
    });
    expect(studio.querySelector('[data-image-studio-icon="true"]')).not.toBeNull();
    expect(edit.querySelector("[data-image-studio-icon]")).toBeNull();
    // The row is compact: no visible word labels on the actions.
    expect(screen.queryByText("Edit")).not.toBeInTheDocument();
    expect(screen.queryByText("Animate")).not.toBeInTheDocument();
    expect(screen.queryByText("Reference")).not.toBeInTheDocument();
  });

  it("offers no actions on a generation preview", () => {
    const preview = imagePart();
    preview.metadata_json = { preview: true };
    render(
      <ArtifactPart
        part={preview}
        origin="generated"
        onEditImage={vi.fn()}
        onAnimateImage={vi.fn()}
      />,
    );

    expect(screen.queryByRole("button", { name: "Edit this image" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Animate this image" })).toBeNull();
  });

  it("backs a picture with a blurred copy of itself, hidden from screen readers", () => {
    // A picture narrower than the card used to sit between flat bars. The
    // backdrop is the same picture, so it must not be announced twice.
    const { container } = render(
      <ArtifactPart
        part={imagePart()}
        origin="generated"
        onEditImage={vi.fn()}
        onAnimateImage={vi.fn()}
      />,
    );

    const backdrop = container.querySelector(".media-backdrop")!;
    const shown = screen.getByRole("img");
    expect(backdrop).toHaveAttribute("aria-hidden", "true");
    expect(backdrop.getAttribute("src")).toBe(shown.getAttribute("src"));
    expect(screen.getAllByRole("img")).toHaveLength(1);
  });
});
