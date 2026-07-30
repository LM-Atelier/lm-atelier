import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { ErrorBoundary } from "./ErrorBoundary";

function Exploding(): never {
  throw new Error("render failed for a synthetic reason");
}

describe("ErrorBoundary", () => {
  // This suite mounts twice; the project cleans up explicitly rather than
  // relying on automatic teardown.
  afterEach(cleanup);

  it("shows a recovery path instead of a blank page when rendering fails", () => {
    // React logs the caught error; keep the suite output readable.
    const logged = vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(
      <ErrorBoundary>
        <Exploding />
      </ErrorBoundary>,
    );

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reload" })).toBeInTheDocument();
    // Local data being safe is the first thing a user needs to know.
    expect(screen.getByText(/stored locally and are not/i)).toBeInTheDocument();
    expect(screen.getByText("render failed for a synthetic reason")).toBeInTheDocument();
    logged.mockRestore();
  });

  it("renders its children untouched when nothing fails", () => {
    render(
      <ErrorBoundary>
        <p>working</p>
      </ErrorBoundary>,
    );

    expect(screen.getByText("working")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
