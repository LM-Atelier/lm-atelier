/** Asking one Studio edit for several results, each its own step beside the others. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioResultCount } from "./StudioResultCount";
import { studioOffersResults } from "./studioApplyPlan";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { useStudioImage } from "./useStudioImage";

vi.mock("./api", () => ({
  api: {
    openStudioSession: vi.fn(),
    studioSession: vi.fn(),
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

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("the result count", () => {
  it("offers one to four results with the current count pressed, and a press chooses", () => {
    const onChange = vi.fn();
    render(<StudioResultCount kind="instruct" value={1} onChange={onChange} />);

    const counts = screen.getByRole("group", { name: "Results" });
    expect(counts).toHaveTextContent("1234");
    expect(screen.getByRole("button", { name: "1" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "3" })).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(screen.getByRole("button", { name: "3" }));

    expect(onChange).toHaveBeenCalledWith(3);
  });

  it("is not offered where every result would be the same, or where one edit is two", () => {
    for (const kind of ["isolate", "background", "subject"]) {
      render(<StudioResultCount kind={kind} value={2} onChange={vi.fn()} />);
      expect(screen.queryByRole("group", { name: "Results" })).toBeNull();
      expect(studioOffersResults(kind)).toBe(false);
      cleanup();
    }
    for (const kind of ["instruct", "brush", "relight", "extend", "enhance", "text"]) {
      expect(studioOffersResults(kind)).toBe(true);
    }
  });
});

function openStudio(tools: unknown[] = []) {
  const session = { id: "chat-studio", messages: [] } as never;
  vi.mocked(api.openStudioSession).mockResolvedValue(session);
  vi.mocked(api.studioSession).mockResolvedValue(session);
  vi.mocked(api.sendTurn).mockResolvedValue({} as never);
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({ tools } as never);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
  vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
}

async function applyWhenReady(name: string) {
  const button = await screen.findByRole("button", { name });
  await waitFor(() => expect(button).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(button);
  await waitFor(() => expect(api.sendTurn).toHaveBeenCalledTimes(1));
}

describe("asking an edit for several results", () => {
  it("sends the count with the edit, and one result as the request always was", async () => {
    openStudio();
    fireEvent.change(await screen.findByRole("textbox"), { target: { value: "Make the sky orange" } });
    fireEvent.click(screen.getByRole("button", { name: "3" }));
    await applyWhenReady("Apply edit");

    expect(api.sendTurn).toHaveBeenCalledWith(
      "chat-studio", "Make the sky orange", "image", ["art-1"], {}, undefined, undefined, undefined, [], 3,
    );

    vi.mocked(api.sendTurn).mockClear();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Make it dusk" } });
    fireEvent.click(screen.getByRole("button", { name: "1" }));
    await applyWhenReady("Apply edit");

    expect(vi.mocked(api.sendTurn).mock.calls[0]).toEqual(
      ["chat-studio", "Make it dusk", "image", ["art-1"], {}, undefined, undefined, undefined],
    );
  });

  it("sends one cutout however many results another tool was set to", async () => {
    openStudio([
      {
        kind: "isolate",
        workflow_class: "matting",
        available: true,
        reason: null,
        workflow_revision_id: "wfrev_cutout",
        adapter_asset_id: null,
      },
    ]);
    fireEvent.click(await screen.findByRole("button", { name: "3" }));
    fireEvent.click(screen.getByRole("button", { name: "Cut the subject out" }));

    expect(screen.queryByRole("group", { name: "Results" })).toBeNull();
    await applyWhenReady("Cut out");

    expect(vi.mocked(api.sendTurn).mock.calls[0]).toHaveLength(8);
  });
});
