import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { GenerationComparisonView } from "./GenerationComparisonView";
import type { GenerationExperimentRequest } from "./generationExperimentTypes";
import type { ArtifactLibraryItem, ModelProfile, WorkflowReadyRevision } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      artifacts: vi.fn(),
      profilesPage: vi.fn(),
      workflowReadyRevisions: vi.fn(),
      workflowRevisionOutputGeometry: vi.fn(),
      preflightGenerationExperiment: vi.fn(),
    },
  };
});

const PICTURE = {
  id: "sha256:" + "e".repeat(64), sha256: "e".repeat(64), kind: "image", media_type: "image/png",
  size_bytes: 1, original_name: "Harbor at dawn", metadata_json: {},
  created_at: "2026-01-01T00:00:00Z", reference_count: 0, chat_ids: [], project_ids: [],
} as ArtifactLibraryItem;
const PROFILE = [{ id: "profile-a", name: "Harbor model", role: "image", engine: "mock" }] as unknown as ModelProfile[];

function row(id: string, name: string): WorkflowReadyRevision {
  return {
    family_id: `family-${id}`, family_name: "Harbor", workflow_id: `workflow-${id}`, workflow_name: name,
    revision_id: id, revision_version: 1,
  } as WorkflowReadyRevision;
}

beforeEach(() => {
  vi.mocked(api.artifacts).mockResolvedValue([PICTURE]);
  vi.mocked(api.profilesPage).mockResolvedValue(PROFILE);
  // Each kind of workflow is offered only where it fits: making from words, or changing a picture.
  vi.mocked(api.workflowReadyRevisions).mockImplementation(async (options = {}) =>
    options.operation === "image_to_image" ? [row("edit-a", "Touch up"), row("edit-b", "Repaint")] : [row("words-a", "Quick")]);
  vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue({ available: false, preset_ids: [] } as never);
  vi.mocked(api.preflightGenerationExperiment).mockReturnValue(new Promise(() => undefined));
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("compares two ways of changing a chosen picture, at that picture's own size", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><GenerationComparisonView /></QueryClientProvider>);
  await screen.findAllByRole("option", { name: "Harbor - Quick · v1" });
  const first = within(screen.getByRole("group", { name: "First choice" }));
  fireEvent.change(first.getByRole("combobox", { name: "Workflow" }), { target: { value: "words-a" } });

  fireEvent.click(screen.getByRole("radio", { name: /A picture from the Media Library/ }));
  fireEvent.click(await screen.findByRole("button", { name: "Harbor at dawn" }));
  fireEvent.click(screen.getByRole("button", { name: "Change this picture" }));

  // A workflow chosen to make pictures from words is not kept for a change.
  await screen.findAllByRole("option", { name: "Harbor - Touch up · v1" });
  expect(first.getByRole("combobox", { name: "Workflow" })).toHaveValue("");
  expect(screen.getAllByRole("option", { name: "Choose a ready image editing workflow" })).toHaveLength(2);
  expect(screen.queryByRole("group", { name: "Size" })).toBeNull();
  expect(screen.getByText("Size: the picture's own, for both choices.")).toBeVisible();

  fireEvent.change(screen.getByRole("textbox", { name: "Prompt" }), { target: { value: "Make the sky warmer" } });
  for (const [group, revision] of [["First choice", "edit-a"], ["Second choice", "edit-b"]] as const) {
    const fields = within(screen.getByRole("group", { name: group }));
    fireEvent.change(fields.getByRole("combobox", { name: "Model" }), { target: { value: "profile-a" } });
    fireEvent.change(fields.getByRole("combobox", { name: "Workflow" }), { target: { value: revision } });
  }
  fireEvent.click(screen.getByRole("button", { name: "Check both choices" }));

  await waitFor(() => expect(api.preflightGenerationExperiment).toHaveBeenCalledTimes(1));
  const sent = vi.mocked(api.preflightGenerationExperiment).mock.calls[0][0] as GenerationExperimentRequest;
  expect(sent).toMatchObject({
    operation: "image_to_image",
    source_artifact_id: PICTURE.id,
    geometry: { mode: "source" },
    arms: [
      { profile_id: "profile-a", workflow_revision_id: "edit-a" },
      { profile_id: "profile-a", workflow_revision_id: "edit-b" },
    ],
  });
});
