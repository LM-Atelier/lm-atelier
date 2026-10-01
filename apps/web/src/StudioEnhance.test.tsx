/** Enhance in the Studio: the size the workflow applies, asked before Apply and sent as it was shown. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import type { EnlargementPreview } from "./studioEnlargement";
import type { SettingField } from "./types";
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
    chatWorkflowSelections: vi.fn(),
    previewEnlargement: vi.fn(),
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
  vi.restoreAllMocks();
});

const WORDS = "Enlarge the picture and restore its detail";

function factor(overrides: Partial<SettingField> = {}): SettingField {
  return {
    key: "upscale_factor", label: "Scale", type: "integer", default: 3, minimum: 1, maximum: 4, step: 1,
    choices: [], scope: "workflow", visibility: "basic", restart_required: false, available: true,
    unavailable_reason: null, help: "", ...overrides,
  };
}

function preview(overrides: Partial<EnlargementPreview> = {}): EnlargementPreview {
  return {
    version: 1, status: "ready", workflow_revision_id: "rev-enlarge", factor: null, fixed_factor: null,
    request_authorized: false, ...overrides,
  };
}

type Answer = EnlargementPreview | Error | "pending";

/** The Studio on one picture, its preview answering each ask in turn and the last answer after that. */
function openEnhance(...answers: Answer[]): QueryClient {
  const session = { id: "chat-studio", messages: [] } as never;
  vi.mocked(api.openStudioSession).mockResolvedValue(session);
  vi.mocked(api.studioSession).mockResolvedValue(session);
  vi.mocked(api.sendTurn).mockResolvedValue({} as never);
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({
    tools: [{ kind: "enhance", workflow_class: "upscale", available: true, reason: null, workflow_revision_id: null, adapter_asset_id: null }],
  } as never);
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
  let asked = 0;
  vi.mocked(api.previewEnlargement).mockImplementation(async () => {
    const answer = answers[Math.min(asked, answers.length - 1)];
    asked += 1;
    if (answer === "pending") return new Promise<never>(() => {});
    if (answer instanceof Error) throw answer;
    return answer;
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
  vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
  return client;
}

async function chooseEnhance() {
  fireEvent.click(await screen.findByRole("button", { name: /^Enlarge and restore detail/ }));
}

async function applyWhenReady(name: string) {
  const button = await screen.findByRole("button", { name });
  await waitFor(() => expect(button).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(button);
  await waitFor(() => expect(api.sendTurn).toHaveBeenCalledTimes(1));
}

/** The workflow scale the slider shows, once it shows it. */
async function showsScale(value: string) {
  await waitFor(() => expect(screen.getByRole("slider", { name: "Workflow scale" })).toHaveValue(value));
}

/** What the enlargement was sent with: its settings, the workflow it named, and that it was one. */
function sent() {
  const call = vi.mocked(api.sendTurn).mock.calls[0];
  return { words: call[1], inputs: call[3], settings: call[4], workflow: call[7], upscale: call[13] };
}

describe("enlarging in the studio", () => {
  it("asks which workflow runs before Apply, and starts at that workflow's own default", async () => {
    openEnhance(preview({ factor: factor() }));
    await chooseEnhance();

    const slider = await screen.findByRole("slider", { name: "Workflow scale" });
    expect(slider).toHaveAttribute("min", "1");
    expect(slider).toHaveAttribute("max", "4");
    expect(slider).toHaveValue("3");
    expect(screen.getByText("Workflow scale").parentElement).toHaveTextContent("Workflow scale 3x");
    expect(api.previewEnlargement).toHaveBeenCalledWith(
      "chat-studio",
      expect.objectContaining({ mode: "image", input_artifact_ids: ["art-1"], upscale: true, text: WORDS, settings: {} }),
      expect.anything(),
    );
    // The scale is the workflow's own setting, which another step may multiply: Apply names no size.
    await applyWhenReady("Enlarge");

    expect(sent()).toEqual({
      words: WORDS, inputs: ["art-1"], settings: { upscale_factor: 3 }, workflow: "rev-enlarge", upscale: true,
    });
  });

  it("sends the factor the person chose within the workflow's range", async () => {
    openEnhance(preview({ factor: factor() }));
    await chooseEnhance();

    fireEvent.change(await screen.findByRole("slider", { name: "Workflow scale" }), { target: { value: "4" } });
    await applyWhenReady("Enlarge");

    expect(sent().settings).toEqual({ upscale_factor: 4 });
  });

  it("offers a number box with only the bounds a field declares, never an invented range", async () => {
    openEnhance(preview({ factor: factor({ minimum: 1, maximum: null, default: 2 }) }));
    await chooseEnhance();

    const box = await screen.findByRole("spinbutton", { name: "Workflow scale" });
    expect(screen.queryByRole("slider")).toBeNull();
    expect(box).toHaveAttribute("min", "1");
    expect(box).not.toHaveAttribute("max");
    fireEvent.change(box, { target: { value: "12" } });
    await applyWhenReady("Enlarge");

    expect(sent().settings).toEqual({ upscale_factor: 12 });
  });

  it("steps only through the whole multiples a whole-number field takes", async () => {
    openEnhance(preview({ factor: factor({ minimum: 1, maximum: 8, multiple_of: 1.5, default: 3, step: 1.5 }) }));
    await chooseEnhance();

    const slider = await screen.findByRole("slider", { name: "Workflow scale" });
    expect(slider).toHaveAttribute("min", "3");
    expect(slider).toHaveAttribute("max", "6");
    expect(slider).toHaveAttribute("step", "3");
    fireEvent.change(slider, { target: { value: "6" } });
    await applyWhenReady("Enlarge");

    expect(sent().settings).toEqual({ upscale_factor: 6 });
  });

  it("sends no factor when the workflow's bounds hold none it would take", async () => {
    openEnhance(preview({ factor: factor({ minimum: 1, maximum: 2, multiple_of: 0.3, default: 1.5, step: null }) }));
    await chooseEnhance();

    expect(await screen.findByText("The workflow sets how much it enlarges.")).toBeInTheDocument();
    expect(screen.queryByRole("slider")).toBeNull();
    expect(screen.queryByRole("spinbutton")).toBeNull();
    await applyWhenReady("Enlarge");

    expect(sent().settings).toEqual({});
  });

  it("offers a workflow's list of factors as choices, never a range between them", async () => {
    openEnhance(preview({ factor: factor({ type: "enum", choices: [2, 4], default: 2, minimum: null, maximum: null, step: null }) }));
    await chooseEnhance();

    const choices = await screen.findByRole("group", { name: "Workflow scale" });
    expect(choices).toHaveTextContent("2x4x");
    expect(screen.getByRole("button", { name: "2x" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("slider")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "4x" }));
    await applyWhenReady("Enlarge");

    expect(sent().settings).toEqual({ upscale_factor: 4 });
  });

  it("sends no factor to a workflow that sets its own size, and says how much when it is known", async () => {
    openEnhance(preview({ fixed_factor: 4 }));
    await chooseEnhance();

    expect(await screen.findByText("Enlarges 4x, set by the workflow.")).toBeInTheDocument();
    expect(screen.queryByRole("slider")).toBeNull();
    await applyWhenReady("Enlarge 4x");

    expect(sent()).toEqual({ words: WORDS, inputs: ["art-1"], settings: {}, workflow: "rev-enlarge", upscale: true });
  });

  it("names no size the workflow does not show", async () => {
    openEnhance(preview());
    await chooseEnhance();

    expect(await screen.findByText("The workflow sets how much it enlarges.")).toBeInTheDocument();
    await applyWhenReady("Enlarge");

    expect(sent().settings).toEqual({});
  });

  it("waits for the answer, and says why when no workflow can take the enlargement", async () => {
    openEnhance("pending");
    await chooseEnhance();

    expect(await screen.findByText("Checking what the workflow can do…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enlarge" })).toHaveAttribute("aria-disabled", "true");

    cleanup();
    openEnhance(new Error("No enlargement workflow is ready."));
    await chooseEnhance();

    expect(await screen.findByText("No enlargement workflow is ready.")).toHaveAttribute("role", "alert");
    fireEvent.click(screen.getByRole("button", { name: "Enlarge" }));
    expect(api.sendTurn).not.toHaveBeenCalled();
  });

  it("asks again when an Apply is refused, and offers no Apply while the new answer is a refusal", async () => {
    openEnhance(preview({ fixed_factor: 2 }), new Error("The enlargement workflow changed."));
    vi.mocked(api.sendTurn).mockRejectedValue(new Error("Refused."));
    await chooseEnhance();
    await applyWhenReady("Enlarge 2x");

    expect(await screen.findByText("The enlargement workflow changed.")).toHaveAttribute("role", "alert");
    expect(api.previewEnlargement).toHaveBeenCalledTimes(2);
    const apply = screen.getByRole("button", { name: "Enlarge" });
    expect(apply).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(apply);
    expect(api.sendTurn).toHaveBeenCalledTimes(1);
  });

  it("asks again when a scoped recipe changes, though the workflow stays the same", async () => {
    const client = openEnhance(preview({ factor: factor() }), preview({ factor: factor({ default: 2 }) }));
    await chooseEnhance();
    await showsScale("3");

    act(() => {
      client.setQueryData(["workflow-recipe-choice", "chat:chat-studio", "image_upscale"], { mode: "preset", preset_id: "neutral" });
    });

    await showsScale("2");
    expect(api.previewEnlargement).toHaveBeenCalledTimes(2);
  });

  it("asks again when an inherited recipe choice changes above the chat", async () => {
    const client = openEnhance(preview({ factor: factor() }), preview({ factor: factor({ default: 2 }) }));
    await chooseEnhance();
    await showsScale("3");

    // A chat that follows the workspace takes the workspace's choice.
    act(() => {
      client.setQueryData(["workflow-recipe-choice", "workspace", "image_upscale"], { mode: "preset", preset_id: "neutral" });
    });

    await showsScale("2");
  });

  it("asks again when the chosen recipe's own settings are edited, under the same recipe", async () => {
    const client = openEnhance(preview({ factor: factor() }), preview({ factor: factor({ default: 2 }) }));
    client.setQueryData(["workflow-recipes"], [{ id: "neutral", use_case: "image_upscale", enabled: true }]);
    await chooseEnhance();
    await showsScale("3");
    const asked = vi.mocked(api.previewEnlargement).mock.calls.length;

    // What the recipe manager does after saving a recipe: the catalog is read again.
    await act(async () => {
      await client.invalidateQueries({ queryKey: ["workflow-recipes"] });
    });

    await showsScale("2");
    expect(vi.mocked(api.previewEnlargement).mock.calls.length).toBeGreaterThan(asked);
  });

  it("asks nothing about an enlargement when an edit made with another tool is refused", async () => {
    openEnhance(preview({ fixed_factor: 2 }));
    vi.mocked(api.sendTurn).mockRejectedValue(new Error("The edit was refused."));
    fireEvent.change(await screen.findByRole("textbox"), { target: { value: "Make the sky orange" } });
    await applyWhenReady("Apply edit");

    await waitFor(() => expect(screen.getByRole("button", { name: "Apply edit" })).toHaveAttribute("aria-disabled", "false"));
    expect(api.previewEnlargement).not.toHaveBeenCalled();
  });
});
