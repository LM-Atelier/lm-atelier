import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { StudioWorkflowSelector } from "./StudioWorkflowSelector";
import { useActiveChatWorkflowSelection } from "./useActiveChatWorkflowSelection";
import type { ActiveChatWorkflowSelectionState } from "./useActiveChatWorkflowSelection";
import type { WorkflowFamily } from "./types";

vi.mock("./useActiveChatWorkflowSelection", () => ({
  useActiveChatWorkflowSelection: vi.fn(),
}));

const editFamily = (overrides: Partial<WorkflowFamily> = {}): WorkflowFamily => ({
  id: "family-edit",
  name: "Krea Identity Edit",
  description: "",
  use_case: "",
  tags: [],
  enabled: true,
  archived: false,
  compatibility: false,
  variants: [{
    id: "variant-edit",
    variant_key: "edit",
    name: "Edit",
    operation: "image_to_image",
    current_revision_id: "revision-edit",
    current_revision_version: 2,
    engine: "comfyui",
    capabilities: ["image"],
    trusted: true,
    readiness: "ready",
    readiness_reason: null,
  }],
  preferences: [{
    selector_capability: "image",
    enabled: true,
    is_default: false,
    sort_order: 0,
  }],
  created_at: "2026-08-09T00:00:00Z",
  updated_at: "2026-08-09T00:00:00Z",
  ...overrides,
});

function readyState(overrides: Partial<Extract<
  ActiveChatWorkflowSelectionState,
  { kind: "ready" }
>> = {}): Extract<ActiveChatWorkflowSelectionState, { kind: "ready" }> {
  return {
    kind: "ready",
    capability: "image",
    choiceKind: "default",
    current: undefined,
    currentFamilyId: null,
    families: [],
    selectedFamilyMissing: false,
    saving: false,
    saveError: null,
    choose: vi.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  vi.mocked(useActiveChatWorkflowSelection).mockReturnValue(readyState());
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Image Studio workflow selection", () => {
  it("offers only families with an image-to-image variant", () => {
    const choose = vi.fn();
    const generationOnly = editFamily({
      id: "family-generate",
      name: "Generate only",
      variants: [{
        ...editFamily().variants[0],
        id: "variant-generate",
        operation: "text_to_image",
      }],
    });
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue(readyState({
      families: [editFamily(), generationOnly],
      choose,
    }));
    const availability = vi.fn();
    const selectionChanged = vi.fn();

    render(
      <StudioWorkflowSelector
        chatId="chat-studio"
        disabled={false}
        onAvailabilityChange={availability}
        onSelectionChange={selectionChanged}
      />,
    );

    expect(screen.getByRole("option", { name: "Krea Identity Edit" })).toBeTruthy();
    expect(screen.queryByRole("option", { name: "Generate only" })).toBeNull();
    fireEvent.change(screen.getByLabelText("Editing workflow"), {
      target: { value: "family-edit" },
    });
    expect(choose).toHaveBeenCalledWith({
      mode: "family",
      workflow_family_id: "family-edit",
    });
    expect(availability).toHaveBeenLastCalledWith("Saving the workflow choice.");
    expect(selectionChanged).toHaveBeenCalledTimes(1);
  });

  it("disables a known-unrunnable edit family and reports its reason", async () => {
    const blocked = editFamily({
      variants: [{
        ...editFamily().variants[0],
        readiness: "setup_required",
        readiness_reason: "model_unavailable",
      }],
    });
    const state = readyState({
      choiceKind: "explicit",
      current: {
        selector_capability: "image",
        mode: "family",
        workflow_family_id: blocked.id,
        workflow_revision_id: null,
        legacy_profile_id: null,
      },
      currentFamilyId: blocked.id,
      families: [blocked],
    });
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue(state);
    const availability = vi.fn();

    render(
      <StudioWorkflowSelector
        chatId="chat-studio"
        disabled={false}
        onAvailabilityChange={availability}
        onSelectionChange={vi.fn()}
      />,
    );

    expect(screen.getByRole("option", { name: /Krea Identity Edit \(not ready\)/i }))
      .toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("The model it needs is not installed.");
    await waitFor(() => expect(availability).toHaveBeenLastCalledWith(
      "The model it needs is not installed.",
    ));
  });

  it("keeps a read failure unknown and offers retry", async () => {
    const retry = vi.fn();
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue({
      kind: "read-error",
      capability: "image",
      error: new Error("workflow service unavailable"),
      retry,
    });
    const availability = vi.fn();

    render(
      <StudioWorkflowSelector
        chatId="chat-studio"
        disabled={false}
        onAvailabilityChange={availability}
        onSelectionChange={vi.fn()}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("workflow service unavailable");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(retry).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(availability).toHaveBeenLastCalledWith(
      "Cannot read the current workflow choice.",
    ));
  });
});
