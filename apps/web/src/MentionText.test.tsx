import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MentionText } from "./MentionText";
import type { MessageReference } from "./types";

function reference(overrides: Partial<MessageReference> = {}): MessageReference {
  return {
    reference_subject_id: "ref-1",
    mention_slug: "ada-lovelace",
    subject_name: "Ada Lovelace",
    subject_kind: "person",
    role: null,
    strength: null,
    source: "mention",
    reference_asset_ids_json: [],
    artifact_ids_json: [],
    ...overrides,
  };
}

function marks(container: HTMLElement) {
  return [...container.querySelectorAll(".message-mention")].map((one) => one.textContent);
}

afterEach(cleanup);

describe("MentionText", () => {
  it("marks a mention the turn actually recorded", () => {
    const { container } = render(
      <MentionText text="draw @ada-lovelace holding a cat" references={[reference()]} />,
    );

    expect(marks(container)).toEqual(["@ada-lovelace"]);
    expect(screen.getByText(/holding a cat/)).toBeTruthy();
  });

  it("marks nothing when the turn recorded nothing", () => {
    // The characters alone are not a reference. Marking them would claim a
    // binding that was never made - the same failure the composer refuses.
    const { container } = render(<MentionText text="draw @ada-lovelace" references={[]} />);

    expect(marks(container)).toEqual([]);
    expect(screen.getByText("draw @ada-lovelace")).toBeTruthy();
  });

  it("shows the name the turn used, not the subject's current one", () => {
    // A rename must not rewrite an old message. The snapshot is the source.
    const { container } = render(
      <MentionText text="draw @ada-lovelace" references={[reference({ subject_name: "Ada Lovelace" })]} />,
    );

    expect(container.querySelector(".message-mention")?.getAttribute("title")).toBe("Ada Lovelace");
  });

  it("marks one occurrence when a chosen mention was also typed again", () => {
    // Two occurrences, one reference. Marking both would assert a second
    // binding that does not exist.
    const { container } = render(
      <MentionText
        text="@ada-lovelace and again @ada-lovelace"
        references={[reference()]}
      />,
    );

    expect(marks(container)).toEqual(["@ada-lovelace"]);
  });

  it("does not mistake a shorter slug for a longer name", () => {
    const { container } = render(
      <MentionText text="draw @ada-lovelace" references={[reference({ mention_slug: "ada" })]} />,
    );

    expect(marks(container)).toEqual([]);
  });

  it("marks each of several recorded subjects once", () => {
    const { container } = render(
      <MentionText
        text="@ada-lovelace beside @grace-hopper"
        references={[
          reference(),
          reference({
            reference_subject_id: "ref-2",
            mention_slug: "grace-hopper",
            subject_name: "Grace Hopper",
          }),
        ]}
      />,
    );

    expect(marks(container)).toEqual(["@ada-lovelace", "@grace-hopper"]);
  });

  it("still marks a subject that has since been deleted", () => {
    // The record outlives the subject, which is when "why does that picture
    // look like that" is most often asked. It links nowhere by design.
    const { container } = render(
      <MentionText text="draw @ada-lovelace" references={[reference()]} />,
    );

    expect(marks(container)).toEqual(["@ada-lovelace"]);
    expect(container.querySelector(".message-mention a")).toBeNull();
  });

  it("leaves an email address alone", () => {
    const { container } = render(
      <MentionText
        text="write to ada@ada-lovelace now"
        references={[reference()]}
      />,
    );

    expect(marks(container)).toEqual([]);
  });
});
