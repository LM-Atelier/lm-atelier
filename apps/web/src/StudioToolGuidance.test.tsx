import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { StudioToolGuidance } from "./StudioToolGuidance";

describe("StudioToolGuidance", () => {
  afterEach(cleanup);

  it("says what is missing and offers the place that fixes it", () => {
    const onOpenWorkflows = vi.fn();
    render(
      <StudioToolGuidance
        reason="Install an inpainting workflow to edit part of a picture."
        onOpenWorkflows={onOpenWorkflows}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent("Install an inpainting workflow");

    fireEvent.click(screen.getByRole("button", { name: "Browse workflows" }));

    // Naming the gap without a way to close it leaves the user exactly where
    // they were, which is what this existed to avoid.
    expect(onOpenWorkflows).toHaveBeenCalledOnce();
  });
});
