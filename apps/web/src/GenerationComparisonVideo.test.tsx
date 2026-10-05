import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ComparisonResults } from "./ComparisonResults";
import { GenerationComparisonView } from "./GenerationComparisonView";
import type { GenerationExperiment, GenerationExperimentRequest } from "./generationExperimentTypes";
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
      generationExperiment: vi.fn(),
      run: vi.fn(),
      jobs: vi.fn(),
    },
  };
});

const STILL = [{ id: "still", name: "Still model", role: "image", engine: "mock" }] as unknown as ModelProfile[];
const MOVING = [{ id: "moving", name: "Motion model", role: "video", engine: "mock" }] as unknown as ModelProfile[];

function row(id: string, name: string): WorkflowReadyRevision {
  return {
    family_id: `family-${id}`, family_name: "Harbor", workflow_id: `workflow-${id}`, workflow_name: name,
    revision_id: id, revision_version: 1,
  } as WorkflowReadyRevision;
}

beforeEach(() => {
  vi.mocked(api.profilesPage).mockImplementation(async (options) => options.role === "video" ? MOVING : STILL);
  // Video workflows are offered only when they are asked for as video ones.
  vi.mocked(api.workflowReadyRevisions).mockImplementation(async (options = {}) =>
    options.operation === "text_to_video" && options.selectorCapability === "video"
      ? [row("clip-a", "Short clip"), row("clip-b", "Long clip")] : [row("still-a", "Quick")]);
  vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue({ available: false, preset_ids: [] } as never);
  vi.mocked(api.preflightGenerationExperiment).mockReturnValue(new Promise(() => undefined));
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("compares two ways of making a video from the same words, at one length and with their choices named", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><GenerationComparisonView /></QueryClientProvider>);
  await screen.findAllByRole("option", { name: "Harbor - Quick · v1" });
  await screen.findAllByRole("option", { name: "Still model" });
  const first = within(screen.getByRole("group", { name: "First choice" }));
  fireEvent.change(first.getByRole("combobox", { name: "Model" }), { target: { value: "still" } });
  fireEvent.change(first.getByRole("combobox", { name: "Workflow" }), { target: { value: "still-a" } });
  fireEvent.click(screen.getByRole("checkbox", { name: /Compare blind/ }));

  fireEvent.click(screen.getByRole("radio", { name: "Words only: make a new video" }));

  // Neither a picture model nor a picture workflow is kept for a video.
  await screen.findAllByRole("option", { name: "Harbor - Short clip · v1" });
  await screen.findAllByRole("option", { name: "Motion model" });
  expect(first.getByRole("combobox", { name: "Model" })).toHaveValue("");
  expect(first.getByRole("combobox", { name: "Workflow" })).toHaveValue("");
  expect(screen.getAllByRole("option", { name: "Choose a ready video workflow" })).toHaveLength(2);
  expect(screen.queryByRole("checkbox", { name: /Compare blind/ })).toBeNull();

  fireEvent.change(screen.getByRole("textbox", { name: "Prompt" }), { target: { value: "Boats drifting at dawn" } });
  fireEvent.change(screen.getByRole("textbox", { name: /Length in seconds/ }), { target: { value: "3" } });
  for (const [group, revision] of [["First choice", "clip-a"], ["Second choice", "clip-b"]] as const) {
    const fields = within(screen.getByRole("group", { name: group }));
    fireEvent.change(fields.getByRole("combobox", { name: "Model" }), { target: { value: "moving" } });
    fireEvent.change(fields.getByRole("combobox", { name: "Workflow" }), { target: { value: revision } });
  }
  fireEvent.click(screen.getByRole("button", { name: "Check both choices" }));

  await waitFor(() => expect(api.preflightGenerationExperiment).toHaveBeenCalledTimes(1));
  const sent = vi.mocked(api.preflightGenerationExperiment).mock.calls[0][0] as GenerationExperimentRequest;
  expect(sent).toMatchObject({
    operation: "text_to_video",
    arms: [
      { profile_id: "moving", workflow_revision_id: "clip-a", settings: { duration_seconds: 3 } },
      { profile_id: "moving", workflow_revision_id: "clip-b", settings: { duration_seconds: 3 } },
    ],
  });
  // Chosen blind for pictures, but a video comparison names its choices.
  expect("evaluation_mode" in sent).toBe(false);
});

it("plays each choice's video once it is made, and says so of videos", async () => {
  vi.mocked(api.generationExperiment).mockResolvedValue({
    id: "gexp-video", name: "Short clip and Long clip", state: "started", operation: "text_to_video",
    source_artifact_id: null, prompt: "Boats drifting at dawn", negative_prompt: "",
    geometry: { mode: "size", width: 512, height: 384 }, seed_policy: { kind: "same_recorded_number", seed: 41 },
    seed_equivalence: "none", preflight_sha256: "a".repeat(64), snapshot_sha256: "b".repeat(64), estimate: [],
    created_at: "2026-10-01T00:00:00Z", work_plan_id: "plan-1", started_at: null,
    evaluation: null, evaluation_mode: "unblinded", blind_pending: false,
    arms: [1, 2].map((ordinal) => ({
      id: `arm-${ordinal}`, ordinal, label: ordinal === 1 ? "Short clip" : "Long clip", outcome: "compatible",
      profile_id: "moving", profile_name: "Motion model", workflow_revision_id: `clip-${ordinal}`, workflow_version: 1,
      workflow_activation_id: null, model_family: null, width: 512, height: 384,
      effective_settings: { frames: ordinal === 1 ? 17 : 41, fps: 8 }, trigger_words_applied: [],
      snapshot_sha256: `${ordinal}`.repeat(64),
      trials: [{ id: `trial-${ordinal}`, ordinal: 1, seed: 41, state: "started", work_step_id: `step-${ordinal}`,
        run_id: `run-${ordinal}`, job_id: `job-${ordinal}`, status: ordinal === 1 ? "complete" : "running" }],
    })),
  } as unknown as GenerationExperiment);
  vi.mocked(api.run).mockResolvedValue({
    id: "run-1", work_step_id: "step-1", status: "complete",
    provenance_json: { generation_experiment: { trial_id: "trial-1" }, outputs: [{ artifact_id: "artifact-1", kind: "video" }] },
  } as never);
  vi.mocked(api.jobs).mockResolvedValue([]);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <ComparisonResults experimentId="gexp-video" onStart={vi.fn()} starting={false} startError={null} onNew={vi.fn()} />
  </QueryClientProvider>);

  const made = await screen.findByLabelText("Made by Short clip");
  expect(made.tagName).toBe("VIDEO");
  expect(made).toHaveAttribute("src", expect.stringContaining("artifact-1"));
  expect(screen.getByText("1 of 2 videos ready")).toBeInTheDocument();
});
