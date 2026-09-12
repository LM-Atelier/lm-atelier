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

/** Beside the picture, not instead of it.
 *
 * The run succeeded and the picture is real and usable; it is simply not the
 * shape that was asked for. So the note sits inside the same card, and the
 * picture, its caption and every action stay exactly where they were.
 */
describe("the size note", () => {
  afterEach(cleanup);

  /** On the PART. Two runs making identical bytes share one artifact, so a
   * judgement stored there would show one conversation the other's answer. */
  function withAgreement(agreement: unknown): MessagePart {
    const base = imagePart();
    return {
      ...base,
      metadata_json: { ...base.metadata_json, output_size_agreement: agreement },
    };
  }

  const disagreed = {
    v: 1,
    state: "disagreed",
    requested_width: 1024,
    requested_height: 768,
    raster_width: 2048,
    raster_height: 1536,
  };

  it("says what arrived and what was asked for, beside the picture", () => {
    render(<ArtifactPart part={withAgreement(disagreed)} origin="generated" />);

    const note = screen.getByRole("status");
    expect(note).toHaveTextContent("This came out 2048 × 1536, not the 1024 × 768 you asked for.");
    // Inside the card, so it reads as being about THIS picture rather than
    // about the conversation.
    expect(note.closest("figure")).toContainElement(screen.getByRole("img", { name: /generated/i }));
  });

  it("leaves the picture and its actions untouched", () => {
    render(
      <ArtifactPart
        part={withAgreement(disagreed)}
        origin="generated"
        onEditImage={vi.fn()}
        onToggleFavorite={vi.fn()}
      />,
    );

    expect(screen.getByRole("img", { name: /generated/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit this image" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Favorite this image" })).toBeInTheDocument();
  });

  it("says nothing when the size was the one asked for", () => {
    render(<ArtifactPart part={withAgreement({ ...disagreed, state: "agreed" })} origin="generated" />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("says nothing when the tool never formed an opinion", () => {
    render(
      <ArtifactPart
        part={withAgreement({ v: 1, state: "not_assessed", reason: "binding_unconfirmed" })}
        origin="generated"
      />,
    );
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("says nothing on a streaming preview, which is not the picture in question", () => {
    const part = withAgreement(disagreed);
    render(
      <ArtifactPart
        part={{ ...part, metadata_json: { ...part.metadata_json, preview: true } }}
        origin="generated"
      />,
    );
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
})
