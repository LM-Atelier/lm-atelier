/** Looks: named starting points for the light and color sliders, readable and changeable on them. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioAdjustTool } from "./StudioAdjustTool";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { ADJUSTMENT_LIMIT, NEUTRAL_ADJUSTMENTS } from "./studioAdjustments";
import { currentLook, lookAdjustments, LOOKS } from "./studioLooks";
import { useStudioImage } from "./useStudioImage";

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

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("the looks", () => {
  it("are each a setting of the ordinary sliders, in whole steps within range", () => {
    expect(new Set(LOOKS.map((look) => look.name)).size).toBe(LOOKS.length);
    for (const look of LOOKS) {
      const set = lookAdjustments(look);
      expect(Object.keys(set).sort()).toEqual(Object.keys(NEUTRAL_ADJUSTMENTS).sort());
      // A look is a setting of the sliders: it leaves the tone curve straight.
      const { curve, ...sliders } = set;
      expect(curve).toEqual([]);
      for (const value of Object.values(sliders)) {
        expect(Number.isInteger(value) && Math.abs(value) <= ADJUSTMENT_LIMIT).toBe(true);
      }
      // A look changes something, and every slider it does not name stays at zero.
      expect(set).not.toEqual(NEUTRAL_ADJUSTMENTS);
      for (const [key, value] of Object.entries(sliders)) {
        if (!(key in look.adjustments)) expect(value).toBe(0);
      }
    }
  });

  it("are recognized on the sliders only while they stand exactly there", () => {
    const mono = LOOKS.find((look) => look.name === "Mono")!;

    expect(currentLook(lookAdjustments(mono))).toBe("Mono");
    expect(currentLook({ ...lookAdjustments(mono), contrast: 11 })).toBeNull();
    expect(currentLook(NEUTRAL_ADJUSTMENTS)).toBeNull();
  });
});

describe("the looks in the light and color panel", () => {
  it("set every slider at once, show the chosen one, and wait while an edit arrives", () => {
    const onLook = vi.fn();
    const panel = (adjustments = NEUTRAL_ADJUSTMENTS, busy = false) => (
      <StudioAdjustTool adjustments={adjustments} busy={busy} onChange={vi.fn()} onReset={vi.fn()}
        onApply={vi.fn()} onLook={onLook} />
    );
    const { rerender } = render(panel());
    const warm = LOOKS.find((look) => look.name === "Warm")!;

    fireEvent.click(screen.getByRole("button", { name: "Warm" }));
    expect(onLook).toHaveBeenCalledExactlyOnceWith(lookAdjustments(warm));
    rerender(panel(lookAdjustments(warm)));
    expect(screen.getByRole("button", { name: "Warm" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Cool" })).toHaveAttribute("aria-pressed", "false");
    rerender(panel(lookAdjustments(warm), true));
    fireEvent.click(screen.getByRole("button", { name: "Cool" }));
    expect(onLook).toHaveBeenCalledTimes(1);
  });

  it("are not offered where nothing would take them", () => {
    render(<StudioAdjustTool adjustments={NEUTRAL_ADJUSTMENTS} busy={false} onChange={vi.fn()}
      onReset={vi.fn()} onApply={vi.fn()} />);

    expect(screen.queryByRole("group", { name: "Looks" })).toBeNull();
  });

  it("applies a chosen look in the studio as the sliders then stand", async () => {
    const session = { id: "chat-studio", messages: [] } as never;
    vi.mocked(api.openStudioSession).mockResolvedValue(session);
    vi.mocked(api.studioSession).mockResolvedValue(session);
    vi.mocked(api.studioLocalEdit).mockResolvedValue(session);
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
    const dramatic = LOOKS.find((look) => look.name === "Dramatic")!;

    fireEvent.click(await screen.findByRole("button", { name: /^Adjust light and color/ }));
    fireEvent.click(screen.getByRole("button", { name: "Dramatic" }));
    // Read on the sliders, and changeable there before Apply.
    expect(screen.getByRole("slider", { name: "Vignette" })).toHaveValue(String(dramatic.adjustments.vignette));
    fireEvent.change(screen.getByRole("slider", { name: "Warmth" }), { target: { value: "5" } });
    expect(screen.getByRole("button", { name: "Dramatic" })).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(screen.getByRole("button", { name: "Apply adjustments" }));

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1", operation: "adjust", adjustments: { ...lookAdjustments(dramatic), warmth: 5 },
    });
  });
});
