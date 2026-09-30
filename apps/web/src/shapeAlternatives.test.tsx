/** Offering the workflows that take a shape, when the chosen one sets its own size. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { OUTPUT_SHAPES_KEY } from "./outputShapePreferences";
import { ShapeAlternativeList } from "./ShapeAlternativeList";
import { shapeAlternativeCandidates, useShapeAlternatives, type ShapeAlternatives } from "./shapeAlternatives";
import type { WorkflowFamily, WorkflowOutputGeometryCapability } from "./types";

vi.mock("./api", () => ({
  api: {
    workflowRevisionOutputGeometry: vi.fn(),
    setChatWorkflowSelection: vi.fn(),
  },
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function family(id: string, overrides: Partial<WorkflowFamily> = {}, operation = "text_to_image"): WorkflowFamily {
  return {
    id,
    name: `Neutral ${id}`,
    description: "",
    use_case: "",
    tags: [],
    enabled: true,
    archived: false,
    compatibility: false,
    variants: [{
      id: `${id}-variant`,
      variant_key: "create",
      name: "Create",
      operation,
      current_revision_id: `${id}-revision`,
      current_revision_version: 1,
      engine: "comfyui",
      capabilities: ["image"],
      trusted: true,
      readiness: "ready",
      readiness_reason: null,
    }],
    preferences: [{ selector_capability: "image", enabled: true, is_default: false, sort_order: 0 }],
    created_at: "2026-09-30T00:00:00Z",
    updated_at: "2026-09-30T00:00:00Z",
    ...overrides,
  };
}

function proof(overrides: Partial<WorkflowOutputGeometryCapability> = {}): WorkflowOutputGeometryCapability {
  return {
    version: 1,
    available: true,
    reason: null,
    revision_id: "revision",
    workflow_id: "workflow",
    artifact_sha256: "a".repeat(64),
    operation: "text_to_image",
    engine: "comfyui",
    size_modes: ["exact", "preset"],
    preset_ids: ["1:1", "16:9"],
    width: null,
    height: null,
    graph_binding_verified: true,
    request_authorized: false,
    ...overrides,
  };
}

describe("the workflows that could answer instead", () => {
  it("are every other ready family for this kind of request, in the picker's order", () => {
    const families = [
      family("b", { preferences: [{ selector_capability: "image", enabled: true, is_default: false, sort_order: 2 }] }),
      family("a", { preferences: [{ selector_capability: "image", enabled: true, is_default: false, sort_order: 1 }] }),
      family("current"),
    ];

    expect(shapeAlternativeCandidates(families, "image", false, "current-revision")).toEqual([
      { familyId: "a", name: "Neutral a", revisionId: "a-revision" },
      { familyId: "b", name: "Neutral b", revisionId: "b-revision" },
    ]);
  });

  it("leave out a family that cannot run this request, or whose revision is not settled until it is sent", () => {
    const notReady = family("not-ready");
    notReady.variants[0] = { ...notReady.variants[0], readiness: "setup_required", readiness_reason: "missing_dependency" };
    const twoWays = family("two-ways");
    twoWays.variants.push({ ...twoWays.variants[0], id: "second", current_revision_id: "second-revision" });
    const families = [
      family("off", { enabled: false }),
      family("archived", { archived: true }),
      family("existing-setup", { compatibility: true }),
      family("video-only", { preferences: [{ selector_capability: "video", enabled: true, is_default: false, sort_order: 0 }] }),
      family("editor", {}, "image_to_image"),
      notReady,
      twoWays,
      family("kept"),
    ];

    expect(shapeAlternativeCandidates(families, "image", false, null).map((one) => one.familyId)).toEqual(["kept"]);
  });

  it("are none for an edit, whose shape comes from its source picture", () => {
    expect(shapeAlternativeCandidates([family("editor", {}, "image_to_image")], "image", true, null)).toEqual([]);
  });

  it("include a video family that starts from a picture", () => {
    const moving = family("moving", {
      preferences: [{ selector_capability: "video", enabled: true, is_default: false, sort_order: 0 }],
    }, "image_to_video");

    expect(shapeAlternativeCandidates([moving], "video", true, null)).toEqual([
      { familyId: "moving", name: "Neutral moving", revisionId: "moving-revision" },
    ]);
  });
});

function renderList(alternatives: Partial<ShapeAlternatives> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const onChoose = vi.fn();
  render(
    <QueryClientProvider client={client}>
      <ShapeAlternativeList alternatives={{
        candidates: [
          { familyId: "shaped", name: "Neutral shaped", revisionId: "shaped-revision" },
          { familyId: "sized", name: "Neutral sized", revisionId: "sized-revision" },
          { familyId: "hidden", name: "Neutral hidden", revisionId: "hidden-revision" },
        ],
        onChoose,
        choosing: false,
        error: null,
        ...alternatives,
      }} />
    </QueryClientProvider>,
  );
  return onChoose;
}

describe("the list of workflows that take a shape", () => {
  afterEach(() => localStorage.clear());

  it("offers only those whose own revision proves a shape that Settings still shows", async () => {
    localStorage.setItem(OUTPUT_SHAPES_KEY, JSON.stringify({
      image: { order: ["1:1", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9"], hidden: ["9:16"] },
    }));
    vi.mocked(api.workflowRevisionOutputGeometry).mockImplementation(async (revisionId) =>
      revisionId === "shaped-revision"
        ? proof()
        : revisionId === "hidden-revision"
          ? proof({ preset_ids: ["9:16"] })
          : proof({ available: false, preset_ids: [] }));

    const onChoose = renderList();

    fireEvent.click(await screen.findByRole("button", { name: "Use Neutral shaped" }));
    expect(onChoose).toHaveBeenCalledExactlyOnceWith("shaped");
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(api.workflowRevisionOutputGeometry).toHaveBeenCalledWith("sized-revision");
  });

  it("says nothing when none of them does", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(proof({ available: false, preset_ids: [] }));

    renderList();

    await waitFor(() => expect(api.workflowRevisionOutputGeometry).toHaveBeenCalledTimes(3));
    expect(screen.queryByText(/take a shape/)).toBeNull();
  });

  it("takes no second choice while the first is being made, and says why a choice failed", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(proof());

    const onChoose = renderList({ choosing: true, error: "Could not change the workflow." });

    const button = await screen.findByRole("button", { name: "Use Neutral shaped" });
    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-disabled", "true");
    expect(onChoose).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("Could not change the workflow.");
  });
});

describe("choosing one for the chat", () => {
  function wrapper(client: QueryClient) {
    return ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }

  it("is offered only where the settings describe the chat's own choice", () => {
    const client = new QueryClient();
    const shown = (enabled: boolean, capability: "image" | "video" | null) => renderHook(() => useShapeAlternatives({
      chatId: "neutral-chat", capability, hasAttachments: false, families: [family("a")], currentRevisionId: null, enabled,
    }), { wrapper: wrapper(client) }).result.current;

    expect(shown(false, "image")).toBeUndefined();
    expect(shown(true, null)).toBeUndefined();
    expect(shown(true, "image")?.candidates.map((one) => one.familyId)).toEqual(["a"]);
  });

  it("chooses the family for this chat's kind of request, as the workflow picker does, and rereads the choice", async () => {
    const client = new QueryClient();
    const reread = vi.spyOn(client, "invalidateQueries");
    vi.mocked(api.setChatWorkflowSelection).mockResolvedValue({
      selector_capability: "image", mode: "family", workflow_family_id: "a", workflow_revision_id: null, legacy_profile_id: null,
    });
    const { result } = renderHook(() => useShapeAlternatives({
      chatId: "neutral-chat", capability: "image", hasAttachments: false, families: [family("a")], currentRevisionId: null, enabled: true,
    }), { wrapper: wrapper(client) });

    act(() => result.current?.onChoose("a"));

    await waitFor(() => expect(reread).toHaveBeenCalledWith({ queryKey: ["chat", "neutral-chat", "workflow-selections"] }));
    expect(api.setChatWorkflowSelection).toHaveBeenCalledExactlyOnceWith(
      "neutral-chat", "image", { mode: "family", workflow_family_id: "a" },
    );
  });
});
