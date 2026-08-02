import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { LineageButton } from "./LineageButton";
import type { EditLineageStep } from "./messageMedia";

const steps: EditLineageStep[] = [
  { artifactId: "artifact-upload", instruction: "Make it a watercolor", messageId: "user-1" },
  { artifactId: "artifact-step-1", instruction: "", messageId: "user-2" },
];

describe("LineageButton", () => {
  afterEach(cleanup);

  it("stays hidden until a result is at least two edits deep", () => {
    render(<LineageButton steps={steps.slice(0, 1)} resultUrl="/api/artifacts/result/content" />);
    expect(screen.queryByRole("button", { name: /Lineage/ })).toBeNull();
  });

  it("walks the chain oldest first and ends at the current result", () => {
    render(<LineageButton steps={steps} resultUrl="/api/artifacts/result/content" />);

    fireEvent.click(screen.getByRole("button", { name: /Lineage/ }));

    const dialog = screen.getByRole("dialog", { name: "Edit lineage" });
    expect(dialog).toBeVisible();
    expect(screen.getByText("2 steps")).toBeInTheDocument();
    expect(screen.getByText("Make it a watercolor")).toBeInTheDocument();
    // A turn with no text still shows as a step, honestly labeled.
    expect(screen.getByText("No written instruction")).toBeInTheDocument();
    const images = screen.getAllByRole("img");
    expect(images[0]).toHaveAttribute("src", "/api/artifacts/artifact-upload/content");
    expect(images.at(-1)).toHaveAttribute("src", "/api/artifacts/result/content");
    expect(screen.getByText("Result")).toBeInTheDocument();
  });
});
