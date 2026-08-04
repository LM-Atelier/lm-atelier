/** Being offered and being the default are two questions, not one. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { WorkflowFamilyPreferences } from "./WorkflowFamilyPreferences";
import { preferenceRefusal } from "./workflowPreferenceErrors";
import type { WorkflowFamily } from "./types";

vi.mock("./api", () => ({ api: { setWorkflowFamilyPreference: vi.fn() } }));

function family(preferences: WorkflowFamily["preferences"] = []): WorkflowFamily {
  return {
    id: "family-1",
    name: "Portrait finish",
    description: "",
    use_case: "image",
    tags: [],
    enabled: true,
    archived: false,
    compatibility: false,
    variants: [],
    preferences,
    created_at: "2026-08-03T00:00:00Z",
    updated_at: "2026-08-03T00:00:00Z",
  };
}

function renderPreferences(one: WorkflowFamily) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <WorkflowFamilyPreferences family={one} />
    </QueryClientProvider>,
  );
}

describe("where a workflow family is offered", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("offers it for a capability without making it the default", async () => {
    vi.mocked(api.setWorkflowFamilyPreference).mockResolvedValue({} as never);
    renderPreferences(family());

    fireEvent.click(screen.getByRole("checkbox", { name: /Images/ }));

    await waitFor(() =>
      expect(api.setWorkflowFamilyPreference).toHaveBeenCalledWith("family-1", "image", {
        enabled: true,
        is_default: false,
        sort_order: 0,
      }),
    );
  });

  it("cannot leave something the default while turning it off", async () => {
    vi.mocked(api.setWorkflowFamilyPreference).mockResolvedValue({} as never);
    renderPreferences(
      family([{ selector_capability: "image", enabled: true, is_default: true, sort_order: 1 }]),
    );

    fireEvent.click(screen.getByRole("checkbox", { name: /Images/ }));

    // The server refuses this combination; agreeing with it beforehand is
    // better than being told off afterwards.
    await waitFor(() =>
      expect(api.setWorkflowFamilyPreference).toHaveBeenCalledWith("family-1", "image", {
        enabled: false,
        is_default: false,
        sort_order: 1,
      }),
    );
  });

  it("only offers the default control where the family is already offered", () => {
    renderPreferences(
      family([{ selector_capability: "image", enabled: true, is_default: false, sort_order: 0 }]),
    );
    expect(screen.getAllByRole("button", { name: "Make this the default" })).toHaveLength(1);
  });

  it("says what the server's refusals mean", () => {
    // "workflow-family-not-selectable" is a code, not a sentence.
    expect(preferenceRefusal("workflow-default-disabled")).toMatch(/before it can be the default/);
    expect(preferenceRefusal("workflow-family-not-selectable")).toMatch(/archived or turned off/);
    expect(preferenceRefusal("something-else")).toBeNull();
  });
});
