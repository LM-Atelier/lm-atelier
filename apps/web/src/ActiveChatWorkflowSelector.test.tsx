import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ActiveChatWorkflowSelector } from "./ActiveChatWorkflowSelector";
import { useActiveChatWorkflowSelection } from "./useActiveChatWorkflowSelection";
import type { WorkflowFamily } from "./types";

vi.mock("./useActiveChatWorkflowSelection", () => ({
  useActiveChatWorkflowSelection: vi.fn(),
}));

function family(overrides: Partial<WorkflowFamily> = {}): WorkflowFamily {
  return {
    id: "family-1",
    name: "Portrait workflow",
    description: "",
    use_case: "",
    tags: [],
    enabled: true,
    archived: false,
    compatibility: false,
    variants: [{
      id: "variant-1",
      variant_key: "create",
      name: "Create",
      operation: "text_to_image",
      current_revision_id: "revision-1",
      current_revision_version: 1,
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
    created_at: "2026-08-08T00:00:00Z",
    updated_at: "2026-08-08T00:00:00Z",
    ...overrides,
  };
}

describe("ActiveChatWorkflowSelector", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("shows classification truth rather than a family selector in routing Auto", () => {
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue({
      kind: "unresolved",
      capability: null,
    });

    render(<ActiveChatWorkflowSelector chatId="chat-1" routingMode="auto" />);

    expect(screen.getByRole("status")).toHaveTextContent("Chosen after request classification");
    expect(screen.queryByRole("combobox")).toBeNull();
  });

  it("offers one Default, Auto, or family choice for an explicit mode", () => {
    const choose = vi.fn();
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue({
      kind: "ready",
      capability: "image",
      choiceKind: "default",
      current: undefined,
      currentFamilyId: null,
      families: [family()],
      selectedFamilyMissing: false,
      saving: false,
      saveError: null,
      choose,
    });

    render(<ActiveChatWorkflowSelector chatId="chat-1" routingMode="image" />);
    const selector = screen.getByRole("combobox", { name: "Workflow for this request type" });
    expect(screen.getAllByRole("combobox")).toHaveLength(1);
    expect(screen.getByRole("option", { name: "Default" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Auto" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Portrait workflow" })).toBeInTheDocument();

    fireEvent.change(selector, { target: { value: "family-1" } });
    expect(choose).toHaveBeenCalledWith({
      mode: "family",
      workflow_family_id: "family-1",
    });
  });

  it("does not claim Default when the current selection cannot be read", () => {
    const retry = vi.fn();
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue({
      kind: "read-error",
      capability: "image",
      error: new Error("selection unavailable"),
      retry,
    });

    render(<ActiveChatWorkflowSelector chatId="chat-1" routingMode="image" />);

    expect(screen.getByRole("combobox")).toBeDisabled();
    expect(screen.getByRole("combobox")).toHaveValue("");
    expect(screen.queryByRole("option", { name: "Default" })).toBeNull();
    const retryButton = screen.getByRole("button", { name: "Try again" });
    expect(retryButton.closest("label")).toBeNull();
    fireEvent.click(retryButton);
    expect(retry).toHaveBeenCalledOnce();
  });

  it("preserves an existing compatibility choice until it is intentionally replaced", () => {
    const choose = vi.fn();
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue({
      kind: "ready",
      capability: "image",
      choiceKind: "compatibility",
      current: {
        selector_capability: "image",
        mode: "legacy",
        workflow_family_id: null,
        workflow_revision_id: null,
        legacy_profile_id: "profile-1",
      },
      currentFamilyId: null,
      families: [family()],
      selectedFamilyMissing: false,
      saving: false,
      saveError: null,
      choose,
    });

    render(<ActiveChatWorkflowSelector chatId="chat-1" routingMode="image" />);

    expect(screen.getByRole("combobox")).toHaveValue("compatibility:legacy");
    expect(screen.getByText(/replaces the existing model setup/)).toBeInTheDocument();
    expect(choose).not.toHaveBeenCalled();
  });

  it("keeps an unavailable selected family visible", () => {
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue({
      kind: "ready",
      capability: "image",
      choiceKind: "explicit",
      current: {
        selector_capability: "image",
        mode: "family",
        workflow_family_id: "family-gone",
        workflow_revision_id: null,
        legacy_profile_id: null,
      },
      currentFamilyId: "family-gone",
      families: [],
      selectedFamilyMissing: true,
      saving: false,
      saveError: null,
      choose: vi.fn(),
    });

    render(<ActiveChatWorkflowSelector chatId="chat-1" routingMode="image" />);
    expect(screen.getByRole("combobox")).toHaveValue("family-gone");
    expect(screen.getByRole("option", { name: "Selected workflow (unavailable)" }))
      .toBeInTheDocument();
  });

  it("explains when every variant in the chosen family is blocked", () => {
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue({
      kind: "ready",
      capability: "image",
      choiceKind: "explicit",
      current: {
        selector_capability: "image",
        mode: "family",
        workflow_family_id: "family-1",
        workflow_revision_id: null,
        legacy_profile_id: null,
      },
      currentFamilyId: "family-1",
      families: [family({
        variants: [{
          ...family().variants[0],
          readiness: "setup_required",
          readiness_reason: "model_unavailable",
        }],
      })],
      selectedFamilyMissing: false,
      saving: false,
      saveError: null,
      choose: vi.fn(),
    });

    render(<ActiveChatWorkflowSelector chatId="chat-1" routingMode="image" />);
    expect(screen.getByRole("status")).toHaveTextContent("The model it needs is not installed.");
  });

  it("does not let a ready unrelated variant hide a blocked active capability", () => {
    const imageVariant = {
      ...family().variants[0],
      readiness: "setup_required" as const,
      readiness_reason: "activation_not_ready",
    };
    const videoVariant = {
      ...family().variants[0],
      id: "variant-video",
      operation: "text_to_video",
      capabilities: ["video"],
      readiness: "ready" as const,
      readiness_reason: null,
    };
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue({
      kind: "ready",
      capability: "image",
      choiceKind: "explicit",
      current: {
        selector_capability: "image",
        mode: "family",
        workflow_family_id: "family-1",
        workflow_revision_id: null,
        legacy_profile_id: null,
      },
      currentFamilyId: "family-1",
      families: [family({ variants: [videoVariant, imageVariant] })],
      selectedFamilyMissing: false,
      saving: false,
      saveError: null,
      choose: vi.fn(),
    });

    render(<ActiveChatWorkflowSelector chatId="chat-1" routingMode="image" />);
    expect(screen.getByRole("status")).toHaveTextContent("Its files are not prepared yet.");
  });

  it("reports a family with no variant for the active capability", () => {
    vi.mocked(useActiveChatWorkflowSelection).mockReturnValue({
      kind: "ready",
      capability: "image",
      choiceKind: "explicit",
      current: {
        selector_capability: "image",
        mode: "family",
        workflow_family_id: "family-1",
        workflow_revision_id: null,
        legacy_profile_id: null,
      },
      currentFamilyId: "family-1",
      families: [family({
        variants: [{
          ...family().variants[0],
          operation: "text_to_video",
          capabilities: ["video"],
        }],
      })],
      selectedFamilyMissing: false,
      saving: false,
      saveError: null,
      choose: vi.fn(),
    });

    render(<ActiveChatWorkflowSelector chatId="chat-1" routingMode="image" />);
    expect(screen.getByRole("status")).toHaveTextContent("no image variant");
  });
});
