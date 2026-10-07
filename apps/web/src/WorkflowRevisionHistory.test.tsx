/** When each workflow revision was created, on the clock somebody chose. */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { CLOCK_KEY } from "./clockPreference";
import type { Workflow } from "./types";
import { WorkflowRevisionHistory } from "./WorkflowRevisionHistory";

afterEach(() => {
  cleanup();
  localStorage.removeItem(CLOCK_KEY);
});

it("writes a revision's creation time on a chosen 24-hour clock", () => {
  localStorage.setItem(CLOCK_KEY, "24");
  const created = new Date(2026, 8, 1, 15, 45).toISOString();
  const workflow = {
    id: "wf-a",
    name: "Neutral workflow",
    current_revision_id: "rev-1",
    revisions: [{ id: "rev-1", version: 1, created_at: created }],
  } as unknown as Workflow;
  render(<WorkflowRevisionHistory workflow={workflow} selectedRevisionId="rev-1" onInspect={() => undefined} />);

  fireEvent.click(screen.getByRole("button", { name: "Show revision history" }));

  const time = screen.getByRole("time");
  expect(time.textContent).toMatch(/15:45/);
  expect(time.textContent).not.toMatch(/\b(AM|PM)\b/i);
});
