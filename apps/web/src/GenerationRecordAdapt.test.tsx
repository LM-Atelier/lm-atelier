import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { GenerationRecordAdapt } from "./GenerationRecordAdapt";
import type { GenerationRecordRequirement, ReplayPlan } from "./generationRecord";
import type { ArtifactLibraryItem, Chat, ModelProfile, WorkflowSummary } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      workflowSummaries: vi.fn(),
      profiles: vi.fn(),
      createChat: vi.fn(),
      deleteChat: vi.fn(),
      adaptGenerationRecord: vi.fn(),
      artifacts: vi.fn(),
    },
  };
});

const CONTENT = new Uint8Array([1, 2, 3]).buffer;
const SHA = "a".repeat(64);

function plan(codes: string[], operation = "text_to_image"): ReplayPlan {
  return { digest: `sha256:${SHA}`, operation, ready: false,
    refusals: codes.map((code) => ({ code, sha256: null, reasons: [] })) };
}

function requirement(kind: GenerationRecordRequirement["kind"], state: GenerationRecordRequirement["state"],
  role: string | null = null): GenerationRecordRequirement {
  return { kind, sha256: "b".repeat(64), role, state };
}

function show(value: ReplayPlan, requirements: GenerationRecordRequirement[] = []) {
  const onStarted = vi.fn();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <GenerationRecordAdapt plan={value} requirements={requirements} content={CONTENT} onStarted={onStarted} />
  </QueryClientProvider>);
  return onStarted;
}

beforeEach(() => {
  vi.mocked(api.workflowSummaries).mockResolvedValue([
    { id: "w1", name: "Harbor workflow", operation: "text_to_image", current_revision_id: "rev_1" },
    { id: "w2", name: "Unpublished", operation: "text_to_image", current_revision_id: null },
  ] as WorkflowSummary[]);
  vi.mocked(api.profiles).mockResolvedValue([
    { id: "profile_image", name: "Picture model", role: "image" },
    { id: "profile_video", name: "Motion model", role: "video" },
  ] as ModelProfile[]);
  vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as Chat);
  vi.mocked(api.deleteChat).mockResolvedValue(undefined as never);
  vi.mocked(api.adaptGenerationRecord).mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("offers nothing when the record itself or its engine stands in the way", () => {
  for (const codes of [["replay-record-incomplete"], ["replay-engine-differs"], ["replay-input-missing", "replay-record-unsupported"]]) {
    show(plan(codes));
    expect(screen.queryByRole("button", { name: "Make a new version" })).toBeNull();
    cleanup();
  }
  show({ ...plan([]), ready: true });
  expect(screen.queryByRole("button", { name: "Make a new version" })).toBeNull();
});

it("makes a new version with the chosen model in a new chat, and shows it", async () => {
  const onStarted = show(plan(["replay-model-missing"]));
  const button = screen.getByRole("button", { name: "Make a new version" });
  expect(button).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(button);
  expect(api.createChat).not.toHaveBeenCalled();

  const model = screen.getByRole("combobox", { name: "Model" });
  await screen.findByRole("option", { name: "Picture model" });
  // Only the models that make this kind of result are offered.
  expect(screen.queryByRole("option", { name: "Motion model" })).toBeNull();
  expect(screen.queryByRole("combobox", { name: "Workflow" })).toBeNull();
  fireEvent.change(model, { target: { value: "profile_image" } });
  fireEvent.click(button);

  await waitFor(() => expect(onStarted).toHaveBeenCalledExactlyOnceWith("chat_new"));
  expect(api.adaptGenerationRecord).toHaveBeenCalledExactlyOnceWith("chat_new", CONTENT,
    { workflowRevisionId: undefined, profileId: "profile_image", loras: [], inputs: [] });
});

it("stands in a chosen workflow and leaves out every LoRA it names", async () => {
  const onStarted = show(plan(["replay-workflow-missing", "replay-lora-missing"]),
    [requirement("lora", "missing"), requirement("lora", "present")]);
  await screen.findByRole("option", { name: "Harbor workflow" });
  expect(screen.queryByRole("option", { name: "Unpublished" })).toBeNull();

  fireEvent.change(screen.getByRole("combobox", { name: "Workflow" }), { target: { value: "rev_1" } });
  expect(screen.getByRole("button", { name: "Make a new version" })).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(screen.getByRole("checkbox", { name: "Leave out its LoRAs" }));
  fireEvent.click(screen.getByRole("button", { name: "Make a new version" }));

  await waitFor(() => expect(onStarted).toHaveBeenCalledOnce());
  expect(api.adaptGenerationRecord).toHaveBeenCalledExactlyOnceWith("chat_new", CONTENT,
    { workflowRevisionId: "rev_1", profileId: undefined, loras: ["0:omit", "1:omit"], inputs: [] });
});

it("says why the choices were refused, and removes the empty chat it made", async () => {
  vi.mocked(api.adaptGenerationRecord).mockRejectedValue(new ApiError(409, "detail", "message", "adaptation-unavailable",
    { refusals: [{ code: "adaptation-model-unusable", kind: "model", sha256: null, reasons: [] }] }));
  const onStarted = show(plan(["replay-model-missing"]));
  await screen.findByRole("option", { name: "Picture model" });
  fireEvent.change(screen.getByRole("combobox", { name: "Model" }), { target: { value: "profile_image" } });

  fireEvent.click(screen.getByRole("button", { name: "Make a new version" }));

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "It cannot be made with these choices: The chosen model cannot be used for this here. Nothing was started.");
  expect(api.deleteChat).toHaveBeenCalledExactlyOnceWith("chat_new");
  expect(onStarted).not.toHaveBeenCalled();
});

it("stands a library picture in for each input picture that is not here", async () => {
  vi.mocked(api.artifacts).mockResolvedValue([
    { id: `sha256:${"c".repeat(64)}`, kind: "image", original_name: "harbor.png" },
  ] as ArtifactLibraryItem[]);
  const onStarted = show(plan(["replay-input-missing"], "image_to_video"),
    [requirement("input", "missing", "source"), requirement("input", "present", "input")]);
  const button = screen.getByRole("button", { name: "Make a new version" });
  expect(screen.getByText("The picture it started from: not here")).toBeInTheDocument();
  expect(screen.queryByText(/Input picture 2/)).toBeNull();
  expect(button).toHaveAttribute("aria-disabled", "true");

  fireEvent.click(screen.getByRole("button", { name: "Choose a picture" }));
  fireEvent.click(await screen.findByRole("button", { name: "harbor.png" }));
  fireEvent.click(screen.getByRole("button", { name: "Use this picture" }));
  expect(await screen.findByText("The picture it started from: harbor.png")).toBeInTheDocument();
  fireEvent.click(button);

  await waitFor(() => expect(onStarted).toHaveBeenCalledOnce());
  expect(api.adaptGenerationRecord).toHaveBeenCalledExactlyOnceWith("chat_new", CONTENT,
    { workflowRevisionId: undefined, profileId: undefined, loras: [], inputs: [`0:sha256:${"c".repeat(64)}`] });
});
