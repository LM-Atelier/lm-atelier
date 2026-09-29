/** A running edit's progress, and the press that stops it. */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioRunningEdit } from "./StudioRunningEdit";
import type { MessagePart } from "./types";

afterEach(() => {
  cleanup();
});

const SAMPLING = {
  id: "progress",
  type: "progress",
  text: "Sampling",
  artifact_id: null,
  metadata_json: { progress: 0.4, phase: "sampling" },
} as unknown as MessagePart;

describe("a running edit in the studio", () => {
  it("shows how far it has got and stops it on a press", () => {
    const onStop = vi.fn();
    render(<StudioRunningEdit part={SAMPLING} stopping={false} onStop={onStop} />);

    expect(screen.getByRole("status")).toHaveTextContent("Sampling");
    fireEvent.click(screen.getByRole("button", { name: "Stop the edit" }));

    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("does not ask again while a stop is on its way", () => {
    const onStop = vi.fn();
    render(<StudioRunningEdit part={SAMPLING} stopping onStop={onStop} />);

    const button = screen.getByRole("button", { name: "Stopping…" });
    fireEvent.click(button);

    expect(button).toHaveAttribute("aria-disabled", "true");
    expect(onStop).not.toHaveBeenCalled();
  });
});
