/** Turning and flipping: a press is the whole edit, and it arrives as the next step. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { useEffect, type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: {
    openStudioSession: vi.fn(),
    studioSession: vi.fn(),
    studioLocalEdit: vi.fn(),
    sendTurn: vi.fn(),
    upload: vi.fn(),
    favoriteArtifact: vi.fn(),
    artifact: vi.fn(),
    editTemplates: vi.fn(),
    studioCapabilities: vi.fn(),
  },
}));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

const stamp = "2026-09-29T00:00:00Z";

type Part = { type: string; text?: string; artifact_id?: string };

function message(id: string, role: "user" | "assistant", parent: string | null, parts: Part[]) {
  return {
    id,
    chat_id: "chat-studio",
    parent_id: parent,
    role,
    status: "complete",
    created_at: stamp,
    updated_at: stamp,
    parts: parts.map((part, position) => ({
      id: `${id}-${position}`,
      position,
      type: part.type,
      text: part.text ?? null,
      artifact_id: part.artifact_id ?? null,
      metadata_json: {},
    })),
  };
}

function session(messages: ReturnType<typeof message>[]) {
  return { id: "chat-studio", messages } as never;
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("turning a picture through the session", () => {
  it("posts the edit and shows the answered session with its new step", async () => {
    const opened = session([]);
    vi.mocked(api.openStudioSession).mockResolvedValue(opened);
    vi.mocked(api.studioSession).mockResolvedValue(opened);
    const turned = session([
      message("m1", "user", null, [{ type: "text", text: "Rotate right" }, { type: "image", artifact_id: "art-source" }]),
      message("m2", "assistant", "m1", [{ type: "image", artifact_id: "art-turned" }]),
    ]);
    vi.mocked(api.studioLocalEdit).mockResolvedValue(turned);
    const { result } = renderHook(() => useStudioSession("art-source", null), { wrapper });
    await waitFor(() => expect(result.current.sessionId).toBe("chat-studio"));

    // What the server holds from here on, whichever read reaches it first.
    vi.mocked(api.studioSession).mockResolvedValue(turned);
    const done = vi.fn();
    result.current.localEdit("rotate_clockwise", "art-source", done);

    await waitFor(() => expect(done).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-source",
      operation: "rotate_clockwise",
    });
    await waitFor(() => expect(result.current.steps.map((step) => step.artifactId)).toEqual(["art-source", "art-turned"]));
    expect(result.current.steps[1].instruction).toBe("Rotate right");
    expect(result.current.steps[1].isSource).toBe(false);
  });
});

function openStudio() {
  vi.mocked(api.openStudioSession).mockResolvedValue(session([]));
  vi.mocked(api.studioSession).mockResolvedValue(session([]));
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
  vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
}

describe("the rotate or flip tool", () => {
  it("turns the picture on screen with one press, with nothing to select or apply", async () => {
    openStudio();
    vi.mocked(api.studioLocalEdit).mockResolvedValue(session([]));
    fireEvent.click(await screen.findByRole("button", { name: /^Rotate or flip/ }));

    expect(screen.queryByRole("button", { name: "Apply edit" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Invert" })).toBeNull();
    expect(screen.queryByRole("textbox", { name: /edit/i })).toBeNull();

    const right = screen.getByRole("button", { name: "Rotate right" });
    await waitFor(() => expect(right).toHaveAttribute("aria-disabled", "false"));
    fireEvent.click(right);

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1",
      operation: "rotate_clockwise",
    });
  });

  it("refuses another press while an edit is still arriving", async () => {
    openStudio();
    vi.mocked(api.studioLocalEdit).mockReturnValue(new Promise(() => {}) as never);
    fireEvent.click(await screen.findByRole("button", { name: /^Rotate or flip/ }));
    const flip = screen.getByRole("button", { name: "Flip horizontally" });
    await waitFor(() => expect(flip).toHaveAttribute("aria-disabled", "false"));
    fireEvent.click(flip);

    await waitFor(() => expect(flip).toHaveAttribute("aria-disabled", "true"));
    expect(screen.getByText("Applying…")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Flip vertically" }));
    expect(api.studioLocalEdit).toHaveBeenCalledTimes(1);
    expect(vi.mocked(api.studioLocalEdit).mock.calls[0][1].operation).toBe("flip_horizontal");
  });
});
