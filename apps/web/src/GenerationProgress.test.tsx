/** A running generation's step and how far along it is. */

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { GenerationProgress } from "./GenerationProgress";
import type { MessagePart } from "./types";

afterEach(() => {
  cleanup();
});

function progress(text: string | null, metadata: Record<string, unknown>): MessagePart {
  return { id: "progress", type: "progress", text, artifact_id: null, metadata_json: metadata } as unknown as MessagePart;
}

describe("generation progress", () => {
  it("names the step and fills the track as far as the work has got", () => {
    render(<GenerationProgress part={progress("Sampling", { progress: 0.4, phase: "sampling" })} />);

    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Sampling");
    expect(status).toHaveAttribute("aria-live", "polite");
    const fill = status.querySelector(".progress-track > div");
    expect(fill).toHaveStyle({ width: "40%" });
    expect(fill).not.toHaveClass("indeterminate");
  });

  it("moves without a width while the amount of work is not known, and says it is working when unnamed", () => {
    render(<GenerationProgress part={progress("", { progress: 0, phase: "queued", indeterminate: true })} />);

    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Working");
    const fill = status.querySelector(".progress-track > div");
    expect(fill).toHaveClass("indeterminate");
    expect(fill).not.toHaveAttribute("style");
  });
});
