import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { GenerationComparisonView } from "./GenerationComparisonView";
import type {
  ArmPreflight,
  GenerationExperiment,
  GenerationExperimentPreflight,
  GenerationExperimentRequest,
} from "./generationExperimentTypes";
import type { ModelProfile, WorkflowReadyRevision } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      profilesPage: vi.fn(),
      workflowReadyRevisions: vi.fn(),
      workflowRevisionOutputGeometry: vi.fn(),
      preflightGenerationExperiment: vi.fn(),
      createGenerationExperiment: vi.fn(),
      generationExperiment: vi.fn(),
      startGenerationExperiment: vi.fn(),
      run: vi.fn(),
      jobs: vi.fn(),
      openBlindView: vi.fn(),
    },
  };
});

const PROFILES = [
  { id: "profile-a", name: "Harbor model", role: "image", engine: "mock" },
  { id: "profile-b", name: "Lighthouse model", role: "image", engine: "mock" },
] as unknown as ModelProfile[];
const WORKFLOWS: WorkflowReadyRevision[] = [
  { family_id: "f1", family_name: "Scenes", workflow_id: "w1", workflow_name: "Quick", revision_id: "revision-a", revision_version: 2, operation: "text_to_image" },
  { family_id: "f2", family_name: "Scenes", workflow_id: "w2", workflow_name: "Careful", revision_id: "revision-b", revision_version: 5, operation: "text_to_image" },
];

function arm(ordinal: number, label: string, overrides: Partial<ArmPreflight> = {}): ArmPreflight {
  return {
    ordinal, label, outcome: "compatible", profile_id: `profile-${ordinal === 1 ? "a" : "b"}`,
    profile_name: ordinal === 1 ? "Harbor model" : "Lighthouse model",
    workflow_revision_id: `revision-${ordinal === 1 ? "a" : "b"}`, workflow_version: ordinal === 1 ? 2 : 5,
    workflow_activation_id: null, model_family: null, width: 1024, height: 1024,
    effective_settings: { steps: ordinal === 1 ? 8 : 20, negative_prompt: "" }, trigger_words_applied: ordinal === 1 ? ["harborlight"] : [],
    snapshot_sha256: `${ordinal}`.repeat(64), ...overrides,
  };
}

function compatible(overrides: Partial<GenerationExperimentPreflight> = {}): GenerationExperimentPreflight {
  return {
    outcome: "compatible", preflight_sha256: "a".repeat(64), seed_equivalence: "none", confirmation_required: false,
    estimate: [
      { resource: "work_units", value: 1000, unit: "work_units", kind: "estimated", source: "admission_formula", confidence: "heuristic" },
      { resource: "output_bytes", value: 3_000_000, unit: "bytes", kind: "estimated", source: "admission_formula", confidence: "heuristic" },
    ],
    arms: [arm(1, "Choice A"), arm(2, "Choice B")], refusals: [], ...overrides,
  };
}

function experiment(state: "ready" | "started", statuses: ("queued" | "complete" | "failed")[] = ["queued", "queued"]): GenerationExperiment {
  return {
    id: "gexp-1", name: "Choice A and Choice B", state, operation: "text_to_image", prompt: "A quiet harbor", negative_prompt: "",
    geometry: { mode: "size", width: 1024, height: 1024 }, seed_policy: { kind: "same_recorded_number", seed: 41 },
    seed_equivalence: "none", preflight_sha256: "a".repeat(64), snapshot_sha256: "b".repeat(64), estimate: compatible().estimate,
    created_at: "2026-10-01T00:00:00Z", work_plan_id: state === "started" ? "plan-1" : null, started_at: null,
    evaluation: null, evaluation_mode: "unblinded", blind_pending: false,
    arms: [1, 2].map((ordinal) => {
      const preflight = arm(ordinal, ordinal === 1 ? "Choice A" : "Choice B");
      return {
        ...preflight, id: `arm-${ordinal}`, width: 1024, height: 1024, snapshot_sha256: preflight.snapshot_sha256 as string,
        trials: [{
          id: `trial-${ordinal}`, ordinal: 1, seed: 41, state: state === "started" ? "started" : "planned",
          work_step_id: state === "started" ? `step-${ordinal}` : null, run_id: state === "started" ? `run-${ordinal}` : null,
          job_id: state === "started" ? `job-${ordinal}` : null, status: state === "started" ? statuses[ordinal - 1] : null,
        }],
      };
    }),
  } as GenerationExperiment;
}

function renderView() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><GenerationComparisonView /></QueryClientProvider>);
}

function choose(group: string, profileId: string, revisionId: string) {
  const fields = within(screen.getByRole("group", { name: group }));
  fireEvent.change(fields.getByRole("combobox", { name: "Model" }), { target: { value: profileId } });
  fireEvent.change(fields.getByRole("combobox", { name: "Workflow" }), { target: { value: revisionId } });
}

async function fillAndCheck() {
  renderView();
  await screen.findAllByRole("option", { name: "Harbor model" });
  await screen.findAllByRole("option", { name: "Scenes - Quick · v2" });
  fireEvent.change(screen.getByRole("textbox", { name: "Prompt" }), { target: { value: "A quiet harbor" } });
  fireEvent.change(screen.getByRole("textbox", { name: "Seed number (optional)" }), { target: { value: "41" } });
  choose("First choice", "profile-a", "revision-a");
  choose("Second choice", "profile-b", "revision-b");
  fireEvent.click(screen.getByRole("button", { name: "Check both choices" }));
}

beforeEach(() => {
  window.sessionStorage.clear();
  vi.mocked(api.profilesPage).mockResolvedValue(PROFILES);
  vi.mocked(api.workflowReadyRevisions).mockImplementation(async (options = {}) =>
    WORKFLOWS.filter((row) => !options.revisionIds?.length || options.revisionIds.includes(row.revision_id)));
  vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue({ available: false, preset_ids: [] } as never);
  vi.mocked(api.jobs).mockResolvedValue([]);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("checking a comparison", () => {
  it("sends the exact request and shows both choices as they would run", async () => {
    vi.mocked(api.preflightGenerationExperiment).mockResolvedValue(compatible());
    await fillAndCheck();
    const heading = await screen.findByRole("heading", { name: "Both choices can run" });
    await waitFor(() => expect(heading).toHaveFocus());
    const sent = vi.mocked(api.preflightGenerationExperiment).mock.calls[0][0] as GenerationExperimentRequest;
    expect(sent).toEqual({
      name: "Choice A and Choice B", operation: "text_to_image", prompt: "A quiet harbor", negative_prompt: "",
      geometry: { mode: "size", width: 1024, height: 1024 }, seed_policy: { kind: "same_recorded_number", seed: 41 },
      arms: [
        { label: "Choice A", profile_id: "profile-a", workflow_revision_id: "revision-a", settings: {} },
        { label: "Choice B", profile_id: "profile-b", workflow_revision_id: "revision-b", settings: {} },
      ],
    });
    expect(screen.getByText("Added to this choice's prompt: harborlight")).toBeInTheDocument();
    expect(screen.getByText("No trigger words added")).toBeInTheDocument();
    const settings = screen.getByRole("table", { name: "Settings each choice runs with" });
    expect(within(settings).getByRole("row", { name: /steps 8 20 Differs/ })).toBeInTheDocument();
    expect(screen.getByText("1,000 work units (estimated)")).toBeInTheDocument();
  });

  it("lists why a comparison cannot run and offers the one alternative a reason names", async () => {
    vi.mocked(api.preflightGenerationExperiment).mockResolvedValue(compatible({
      outcome: "refused", preflight_sha256: null, arms: [arm(1, "Choice A"), arm(2, "Choice B", { outcome: "refused" })],
      refusals: [
        { code: "arm-workflow-untrusted", arm_ordinal: 2, setting: null, alternative: null, message: "This workflow has not been reviewed." },
        { code: "seed-family-unproven", arm_ordinal: null, setting: null,
          alternative: { seed_policy: "same_recorded_number", profile_id: null }, message: "Use the same recorded number instead." },
      ],
    }));
    await fillAndCheck();
    await screen.findByRole("heading", { name: "This comparison cannot run as asked" });
    expect(screen.getByText("Choice B")).toBeInTheDocument();
    expect(screen.getByText("Whole comparison")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Accept this comparison" })).toBeNull();
    fireEvent.click(screen.getByRole("radio", { name: "Same starting noise (one model family only)" }));
    fireEvent.click(screen.getByRole("button", { name: "Use “Same number for both” instead" }));
    expect(screen.getByRole("radio", { name: "Same number for both" })).toBeChecked();
  });

  it("asks for a blind comparison when one is chosen", async () => {
    vi.mocked(api.preflightGenerationExperiment).mockResolvedValue(compatible());
    renderView();
    fireEvent.click(screen.getByRole("checkbox", { name: /Compare blind/ }));
    await screen.findAllByRole("option", { name: "Harbor model" });
    await screen.findAllByRole("option", { name: "Scenes - Quick · v2" });
    fireEvent.change(screen.getByRole("textbox", { name: "Prompt" }), { target: { value: "A quiet harbor" } });
    choose("First choice", "profile-a", "revision-a");
    choose("Second choice", "profile-b", "revision-b");
    fireEvent.click(screen.getByRole("button", { name: "Check both choices" }));
    await waitFor(() => expect(api.preflightGenerationExperiment).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.preflightGenerationExperiment).mock.calls[0][0]).toMatchObject({ evaluation_mode: "blind" });
  });

  it("refuses to send a draft it can already tell is incomplete", async () => {
    renderView();
    fireEvent.click(screen.getByRole("button", { name: "Check both choices" }));
    expect(await screen.findByText("Choose an image model for each choice.")).toBeInTheDocument();
    expect(api.preflightGenerationExperiment).not.toHaveBeenCalled();
  });
});

describe("accepting and starting", () => {
  it("does not accept a check the draft has moved away from", async () => {
    vi.mocked(api.preflightGenerationExperiment).mockResolvedValue(compatible());
    await fillAndCheck();
    await screen.findByRole("heading", { name: "Both choices can run" });
    fireEvent.change(screen.getByRole("textbox", { name: "Prompt" }), { target: { value: "A different harbor" } });
    expect(screen.getByText("Changed since it was checked. Check again before accepting.")).toBeInTheDocument();
    const accept = screen.getByRole("button", { name: "Accept this comparison" });
    expect(accept).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(accept);
    expect(api.createGenerationExperiment).not.toHaveBeenCalled();
  });

  it("accepts exactly what was checked, retries with the same key, then starts both pictures", async () => {
    vi.mocked(api.preflightGenerationExperiment).mockResolvedValue(compatible());
    vi.mocked(api.createGenerationExperiment)
      .mockRejectedValueOnce(new TypeError("Network down"))
      .mockResolvedValueOnce(experiment("ready"));
    vi.mocked(api.generationExperiment).mockResolvedValue(experiment("ready"));
    vi.mocked(api.startGenerationExperiment).mockResolvedValue(experiment("started"));
    await fillAndCheck();
    fireEvent.click(await screen.findByRole("button", { name: "Accept this comparison" }));
    await screen.findByText("Network down");
    fireEvent.click(screen.getByRole("button", { name: "Accept this comparison" }));
    await screen.findByRole("button", { name: "Make both pictures" });
    const [first, second] = vi.mocked(api.createGenerationExperiment).mock.calls.map(([payload]) => payload);
    expect(second.idempotency_key).toBe(first.idempotency_key);
    expect(second.preflight_sha256).toBe("a".repeat(64));
    expect(second.prompt).toBe("A quiet harbor");
    expect(window.sessionStorage.getItem("lm-atelier.generation-comparison")).toBe("gexp-1");
    vi.mocked(api.generationExperiment).mockResolvedValue(experiment("started"));
    fireEvent.click(screen.getByRole("button", { name: "Make both pictures" }));
    await waitFor(() => expect(api.startGenerationExperiment).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.startGenerationExperiment).mock.calls[0]).toEqual(["gexp-1", expect.objectContaining({
      snapshot_sha256: "b".repeat(64), confirm_expensive: false,
    })]);
    expect(await screen.findByText("0 of 2 pictures ready")).toBeInTheDocument();
    // Nothing to prefer until a picture is there.
    expect(screen.queryByRole("group", { name: "Which do you prefer?" })).toBeNull();
  });

  it("shows the reasons a refused accept carries", async () => {
    vi.mocked(api.preflightGenerationExperiment).mockResolvedValue(compatible());
    vi.mocked(api.createGenerationExperiment).mockRejectedValue(new ApiError(422, "x",
      "A choice in this comparison cannot run as asked. Check it again for the reasons.", "generation-experiment-refused",
      { refusals: [{ code: "arm-profile-unavailable", arm_ordinal: 1, setting: null, alternative: null, message: "This model is not available for pictures." }] }));
    await fillAndCheck();
    fireEvent.click(await screen.findByRole("button", { name: "Accept this comparison" }));
    expect(await screen.findByText(/This model is not available for pictures\./)).toBeInTheDocument();
    expect(screen.getByText("Choice A")).toBeInTheDocument();
  });

  it("asks before making large pictures and sends nothing when declined", async () => {
    vi.mocked(api.preflightGenerationExperiment).mockResolvedValue(compatible({ confirmation_required: true }));
    vi.mocked(api.createGenerationExperiment).mockResolvedValue(experiment("ready"));
    vi.mocked(api.generationExperiment).mockResolvedValue(experiment("ready"));
    vi.mocked(api.startGenerationExperiment).mockResolvedValue(experiment("started"));
    await fillAndCheck();
    fireEvent.click(await screen.findByRole("button", { name: "Accept this comparison" }));
    fireEvent.click(await screen.findByRole("button", { name: "Make both pictures" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(api.startGenerationExperiment).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Make both pictures" }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Make both pictures" }));
    await waitFor(() => expect(api.startGenerationExperiment).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.startGenerationExperiment).mock.calls[0][1].confirm_expensive).toBe(true);
  });
});

describe("a remembered comparison", () => {
  it("reopens it and shows each finished picture from its own run", async () => {
    window.sessionStorage.setItem("lm-atelier.generation-comparison", "gexp-1");
    vi.mocked(api.generationExperiment).mockResolvedValue(experiment("started", ["complete", "failed"]));
    vi.mocked(api.run).mockResolvedValue({
      id: "run-1", work_step_id: "step-1", status: "complete",
      provenance_json: { generation_experiment: { trial_id: "trial-1" }, outputs: [{ artifact_id: "artifact-1", kind: "image" }] },
    } as never);
    renderView();
    expect(await screen.findByRole("img", { name: "Made by Choice A" })).toHaveAttribute("src", expect.stringContaining("artifact-1"));
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("1 of 2 pictures ready")).toBeInTheDocument();
    expect(api.run).toHaveBeenCalledTimes(1);
    // Only the choice whose picture is finished can be kept as a recipe.
    const kept = screen.getAllByRole("button", { name: "Keep as a recipe" });
    expect(kept).toHaveLength(1);
    expect(kept[0].closest("section")).toHaveAccessibleName("Choice A");
    expect(screen.getByRole("group", { name: "Which do you prefer?" })).toBeInTheDocument();
  });

  it("shows a blind comparison only by position, and reads nothing that names a picture's choice", async () => {
    window.sessionStorage.setItem("lm-atelier.generation-comparison", "gexp-1");
    const named = experiment("started", ["complete", "complete"]);
    vi.mocked(api.generationExperiment).mockResolvedValue({
      ...named, evaluation_mode: "blind", blind_pending: true,
      arms: named.arms.map((arm) => ({ ...arm, trials: arm.trials.map((trial) => ({
        ...trial, work_step_id: null, run_id: null, job_id: null, status: null,
      })) })),
    });
    vi.mocked(api.openBlindView).mockResolvedValue({ id: "gview-1", experiment_id: "gexp-1", evaluation: null, reveal: null,
      positions: [{ position: 1, status: "complete", ready: true }, { position: 2, status: "complete", ready: true }] });
    renderView();
    expect(await screen.findByRole("region", { name: "Compared blind" })).toBeInTheDocument();
    expect(await screen.findByRole("img", { name: "Shown at position 1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Prefer picture 2" })).toBeInTheDocument();
    // No picture is shown under its choice, and nothing leads from one to the other.
    expect(screen.queryByRole("img", { name: /Made by/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /Prefer Choice/ })).toBeNull();
    expect(screen.queryByRole("button", { name: "Keep as a recipe" })).toBeNull();
    expect(screen.queryByText(/pictures ready|Both pictures are ready/, { selector: ".comparison-results > p" })).toBeNull();
    expect(api.run).not.toHaveBeenCalled();
    expect(api.jobs).not.toHaveBeenCalled();
  });

  it("goes back to a new comparison when the remembered one is gone", async () => {
    window.sessionStorage.setItem("lm-atelier.generation-comparison", "gexp-1");
    vi.mocked(api.generationExperiment).mockRejectedValue(new ApiError(404, "x", "This comparison no longer exists.", "generation-experiment-not-found"));
    renderView();
    expect(await screen.findByText("This comparison no longer exists.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "New comparison" }));
    expect(await screen.findByRole("button", { name: "Check both choices" })).toBeInTheDocument();
    expect(window.sessionStorage.getItem("lm-atelier.generation-comparison")).toBeNull();
  });
});
