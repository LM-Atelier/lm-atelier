import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ActiveChatWorkflowSelector } from "./ActiveChatWorkflowSelector";
import { StudioWorkflowSelector } from "./StudioWorkflowSelector";
import { WorkflowSelector } from "./WorkflowSelector";
import type { WorkflowFamily } from "./types";

vi.mock("./api", () => ({ api: {
  workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(),
  projectWorkflowSelections: vi.fn(), setChatWorkflowSelection: vi.fn(),
  setProjectWorkflowSelection: vi.fn(),
} }));

afterEach(() => { cleanup(); vi.resetAllMocks(); });

function family(id: string): WorkflowFamily {
  return {
    id, name: id, description: "", use_case: "", tags: [], enabled: true,
    archived: false, compatibility: false, created_at: "2026-09-30", updated_at: "2026-09-30",
    preferences: [{ selector_capability: "image", enabled: true, is_default: false, sort_order: 0 }],
    variant_count: 3, ready_variant_count: 1, best_readiness: "ready",
    variants: [{ id: id + "-edit", name: "Edit", variant_key: "edit", operation: "image_to_image",
      current_revision_id: null, current_revision_version: null, engine: null,
      capabilities: ["image"], trusted: false, readiness: "setup_required",
      readiness_reason: "Needs installation" }],
  };
}

function mount(kind: "chat" | "project" | "composer" | "studio", chosen = "Chosen elsewhere") {
  const selection = [{ selector_capability: "image" as const, mode: "family" as const,
    workflow_family_id: chosen, workflow_revision_id: null, legacy_profile_id: null }];
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue(selection);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue(selection);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const availability = vi.fn();
  render(<QueryClientProvider client={client}>
    {kind === "studio" ? <StudioWorkflowSelector chatId="chat" disabled={false}
      onAvailabilityChange={availability} onSelectionChange={vi.fn()} />
      : kind === "composer" ? <ActiveChatWorkflowSelector chatId="chat" routingMode="image" />
        : <WorkflowSelector scope={kind} scopeId="scope" capability="image" label="Images" />}
  </QueryClientProvider>);
  return { client, availability };
}

it.each(["chat", "project", "composer", "studio"] as const)(
  "%s preserves an off-page choice through search and uses all relevant variants for readiness",
  async kind => {
    vi.mocked(api.workflowFamilies).mockImplementation(async (_cap, _archived, _dependencies, options) => {
      if (options?.familyIds) return [family("Chosen elsewhere")];
      if (options?.search) return [family("Found by search")];
      return Array.from({ length: 50 }, (_, i) => family(`Page ${i}`));
    });
    const { availability } = mount(kind);
    const chosen = await screen.findByRole("option", { name: "Chosen elsewhere" });
    expect(chosen).toBeEnabled();
    expect(screen.getByRole("combobox")).toHaveValue("Chosen elsewhere");
    expect(screen.queryByText("Needs installation")).not.toBeInTheDocument();
    if (kind === "studio") await waitFor(() => expect(availability).toHaveBeenLastCalledWith(null));
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "Found" } });
    await screen.findByRole("option", { name: "Found by search" });
    expect(screen.getByRole("combobox")).toHaveValue("Chosen elsewhere");
    expect(screen.getByRole("option", { name: "Chosen elsewhere" })).toBeEnabled();
    expect(api.workflowFamilies).toHaveBeenCalledWith("image", false, false,
      expect.objectContaining({ limit: 50, variantLimit: 1, variantCapability: "image", search: "Found",
        ...(kind === "studio" ? { operation: "image_to_image" } : {}) }), expect.any(AbortSignal));
    expect(api.setChatWorkflowSelection).not.toHaveBeenCalled();
    expect(api.setProjectWorkflowSelection).not.toHaveBeenCalled();
  },
);

it("keeps the chosen family when another page fails and retries that page", async () => {
  let fail = true;
  vi.mocked(api.workflowFamilies).mockImplementation(async (_cap, _archived, _dependencies, options) => {
    if (options?.familyIds) return [family("Chosen elsewhere")];
    if (options?.offset === 50) {
      if (fail) throw new Error("Another page failed");
      return [family("Later choice")];
    }
    return Array.from({ length: 50 }, (_, i) => family(`Page ${i}`));
  });
  mount("composer");
  await screen.findByRole("option", { name: "Chosen elsewhere" });
  fireEvent.click(screen.getByRole("button", { name: "Load more image workflows" }));
  await screen.findByText("Another page failed");
  expect(screen.getByRole("combobox")).toHaveValue("Chosen elsewhere");
  fail = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry image workflows" }));
  await screen.findByRole("option", { name: "Later choice" });
  expect(screen.getByRole("combobox")).toHaveValue("Chosen elsewhere");
});

it("does not restore a removed selection from a stale family page", async () => {
  vi.mocked(api.workflowFamilies).mockImplementation(async (_cap, _archived, _dependencies, options) =>
    options?.familyIds ? [] : [family("Chosen elsewhere")]);
  mount("chat");
  await screen.findByRole("option", { name: "Selected workflow (unavailable)" });
  expect(screen.queryByRole("option", { name: "Chosen elsewhere" })).not.toBeInTheDocument();
  expect(screen.getByRole("combobox")).toHaveValue("Chosen elsewhere");
});

it("keeps a failed exact selected-family read distinct from a removed family", async () => {
  vi.mocked(api.workflowFamilies).mockImplementation(async (_cap, _archived, _dependencies, options) => {
    if (options?.familyIds) throw new Error("Selected family read failed");
    return [family("Chosen elsewhere")];
  });
  mount("composer");
  await screen.findByText("Selected family read failed");
  expect(screen.getByRole("combobox")).toBeDisabled();
  expect(screen.queryByRole("option", { name: "Selected workflow (unavailable)" })).not.toBeInTheDocument();
});

it("shows a pending exact family read as loading rather than unavailable", async () => {
  let finish!: (families: WorkflowFamily[]) => void;
  vi.mocked(api.workflowFamilies).mockImplementation(async (_cap, _archived, _dependencies, options) => {
    if (options?.familyIds) return new Promise<WorkflowFamily[]>(resolve => { finish = resolve; });
    return [family("First page")];
  });
  mount("project");
  await waitFor(() => expect(finish).toBeTypeOf("function"));
  expect(screen.getByRole("option", { name: "Loading current choice…" })).toBeInTheDocument();
  expect(screen.queryByRole("option", { name: "Selected workflow (unavailable)" })).not.toBeInTheDocument();
  expect(screen.getByRole("combobox")).toBeDisabled();
  finish([family("Chosen elsewhere")]);
  await screen.findByRole("option", { name: "Chosen elsewhere" });
  expect(screen.getByRole("combobox")).toHaveValue("Chosen elsewhere");
});
