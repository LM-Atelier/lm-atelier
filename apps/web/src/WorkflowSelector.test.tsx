/** Choosing which workflow answers a kind of request. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { WorkflowSelector } from "./WorkflowSelector";
import { orderFamilies, servesCapability } from "./workflowFamilies";
import type { WorkflowFamily, WorkflowFamilyVariant } from "./types";

vi.mock("./api", () => ({
  api: {
    workflowFamilies: vi.fn(),
    chatWorkflowSelections: vi.fn(),
    setChatWorkflowSelection: vi.fn(),
    projectWorkflowSelections: vi.fn(),
    setProjectWorkflowSelection: vi.fn(),
  },
}));

function variant(overrides: Partial<WorkflowFamilyVariant> = {}): WorkflowFamilyVariant {
  return {
    id: "variant-1",
    variant_key: "text_to_image",
    name: "Text to image",
    operation: "text_to_image",
    current_revision_id: "rev-1",
    current_revision_version: 3,
    engine: "comfyui",
    capabilities: ["image"],
    trusted: true,
    readiness: "ready",
    readiness_reason: null,
    ...overrides,
  };
}

function family(overrides: Partial<WorkflowFamily> = {}): WorkflowFamily {
  return {
    id: "family-1",
    name: "Portrait finish",
    description: "",
    use_case: "image",
    tags: [],
    enabled: true,
    archived: false,
    compatibility: false,
    variants: [variant()],
    preferences: [
      { selector_capability: "image", enabled: true, is_default: false, sort_order: 1 },
    ],
    created_at: "2026-08-03T00:00:00Z",
    updated_at: "2026-08-03T00:00:00Z",
    ...overrides,
  };
}

function renderSelector(scope: "chat" | "project" = "chat") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <WorkflowSelector scope={scope} scopeId="scope-1" capability="image" label="Images" />
    </QueryClientProvider>,
  );
}

describe("WorkflowSelector", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("offers only families that serve the capability being chosen for", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([
      family(),
      family({
        id: "family-2",
        name: "Video finish",
        preferences: [
          { selector_capability: "video", enabled: true, is_default: false, sort_order: 1 },
        ],
      }),
      family({
        id: "family-3",
        name: "Turned off here",
        preferences: [
          { selector_capability: "image", enabled: false, is_default: false, sort_order: 2 },
        ],
      }),
    ]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
    renderSelector();

    expect(await screen.findByRole("option", { name: "Portrait finish" })).toBeInTheDocument();
    // A family that serves video, or one disabled for this capability, is not
    // a choice here - offering it would be offering something that cannot run.
    expect(screen.queryByRole("option", { name: "Video finish" })).toBeNull();
    expect(screen.queryByRole("option", { name: "Turned off here" })).toBeNull();
  });

  it("sends the family the user picked", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([family()]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
    vi.mocked(api.setChatWorkflowSelection).mockResolvedValue({
      selector_capability: "image",
      mode: "family",
      workflow_family_id: "family-1",
      workflow_revision_id: null,
      legacy_profile_id: null,
    });
    renderSelector();

    await screen.findByRole("option", { name: "Portrait finish" });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "family-1" } });

    await waitFor(() =>
      expect(api.setChatWorkflowSelection).toHaveBeenCalledWith("scope-1", "image", {
        mode: "family",
        workflow_family_id: "family-1",
      }),
    );
  });

  it("says what following the level above means in each scope", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
    vi.mocked(api.projectWorkflowSelections).mockResolvedValue([]);

    renderSelector("chat");
    expect(await screen.findByRole("option", { name: "Use the project's choice" })).toBeInTheDocument();
    cleanup();

    renderSelector("project");
    // Same idea, two words for it - a chat defaults, a project inherits.
    expect(
      await screen.findByRole("option", { name: "Use the workspace default" }),
    ).toBeInTheDocument();
  });

  it("shows an exact project revision as the active compatibility choice", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([]);
    vi.mocked(api.projectWorkflowSelections).mockResolvedValue([
      {
        selector_capability: "image",
        mode: "revision",
        workflow_family_id: null,
        workflow_revision_id: "rev-legacy",
        legacy_profile_id: null,
      },
    ]);
    renderSelector("project");

    await waitFor(() =>
      expect(screen.getByRole("combobox")).toHaveValue("compatibility:revision"),
    );
    expect(screen.getByRole("option", { name: "Exact workflow revision (existing choice)" }))
      .toBeInTheDocument();
    expect(screen.getByText(/Pinned to one exact revision/)).toBeInTheDocument();
  });

  it("shows a legacy chat profile instead of claiming the project is active", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([
      {
        selector_capability: "image",
        mode: "legacy",
        workflow_family_id: null,
        workflow_revision_id: null,
        legacy_profile_id: "profile-legacy",
      },
    ]);
    renderSelector();

    await waitFor(() =>
      expect(screen.getByRole("combobox")).toHaveValue("compatibility:legacy"),
    );
    expect(screen.getByRole("option", { name: "Existing model setup" })).toBeInTheDocument();
    expect(screen.getByText(/Using the model previously configured here/)).toBeInTheDocument();
  });

  it("keeps an unavailable selected family visible", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([
      {
        selector_capability: "image",
        mode: "family",
        workflow_family_id: "family-missing",
        workflow_revision_id: null,
        legacy_profile_id: null,
      },
    ]);
    renderSelector();

    await waitFor(() => expect(screen.getByRole("combobox")).toHaveValue("family-missing"));
    expect(screen.getByRole("option", { name: "Selected workflow (unavailable)" }))
      .toBeInTheDocument();
  });

  it("warns when the chosen family cannot run at all", async () => {
    const blocked = family({
      variants: [
        variant({ readiness: "setup_required", readiness_reason: "Two model files are missing." }),
      ],
    });
    vi.mocked(api.workflowFamilies).mockResolvedValue([blocked]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([
      {
        selector_capability: "image",
        mode: "family",
        workflow_family_id: "family-1",
        workflow_revision_id: null,
        legacy_profile_id: null,
      },
    ]);
    renderSelector();

    // Letting someone pick a workflow and only telling them at generation
    // time moves the failure rather than preventing it.
    expect(await screen.findByText("Two model files are missing.")).toBeInTheDocument();
  });

  it("stays quiet when only some variants are blocked", () => {
    const partly = family({
      variants: [variant(), variant({ id: "v2", readiness: "unavailable" })],
    });
    // The family can still answer the request through its ready variant, so
    // a warning here would be crying wolf.
    expect(servesCapability(partly, "image")).toBe(true);
  });

  it("orders families by the capability's own sort order", () => {
    const first = family({ id: "a", name: "Zebra", preferences: [
      { selector_capability: "image", enabled: true, is_default: false, sort_order: 1 },
    ] });
    const second = family({ id: "b", name: "Aardvark", preferences: [
      { selector_capability: "image", enabled: true, is_default: false, sort_order: 2 },
    ] });
    // Sort order belongs to the preference, not the family: the same family
    // can sit anywhere in two different capabilities' lists.
    expect(orderFamilies([second, first], "image").map((one) => one.id)).toEqual(["a", "b"]);
  });
});
