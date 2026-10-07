import { useState, type ReactNode } from "react";
import { SENSITIVE_MEDIA_KEY } from "./sensitiveMedia";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import type { ComposerDraft } from "./composerPromptSource";
import { TurnEditor, type TurnEditorProps, type TurnEditorState, type TurnEditorSubmission } from "./TurnEditor";
import type { Artifact, ChatDetail, EngineCapabilities, EngineRole, GenerationPreset } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, profilesPage: vi.fn().mockResolvedValue([]),
    presetsPage: vi.fn(async (options) => (await import("./test/modelLibraryPageFixtures")).presetPages({ presets: async () => presets }, options)),
    workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(), projectWorkflowSelections: vi.fn(),
    classifyDraft: vi.fn(), references: vi.fn(), upload: vi.fn(), setChatWorkflowSelection: vi.fn(),
    workflowRevisionSourceFit: vi.fn(), previewWorkflowRevisionSourceFit: vi.fn(), previewTurnSourceFit: vi.fn(),
    previewPriorTurnSourceFit: vi.fn(),
  } };
});

vi.mock("./LibraryAttachPicker", () => ({
  LibraryAttachPicker: ({ onAttach, onClose }: {
    onAttach: (attachment: { id: string; kind: "image"; origin: "generated" }) => void;
    onClose: () => void;
  }) => <button onClick={() => { onAttach({ id: "library-image", kind: "image", origin: "generated" }); onClose(); }}>Choose library image</button>,
}));

const stamp = "2026-09-06T12:00:00Z";
const chat: ChatDetail = {
  id: "chat-editor", project_id: null, title: "Editor example", archived: false, pinned: false,
  routing_mode: "text", confirm_uncertain_media: false, active_chat_profile_id: null,
  active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: null,
  created_at: stamp, updated_at: stamp, messages: [],
};
const engine: EngineCapabilities = {
  engine: "mock", version: "1", roles: ["chat", "image", "video"], operations: [], formats: [], devices: [],
  streaming: false, tool_calling: false, healthy: true, details: {},
  settings: [{ key: "width", label: "Width", type: "integer", default: 512, minimum: 64, maximum: 2048,
    step: 64, choices: [], scope: "request", visibility: "basic", restart_required: false,
    available: true, unavailable_reason: null, help: "" }],
};
const presets: GenerationPreset[] = [{ id: "preset-one", name: "Landscape", role: "image", settings_json: { width: 768 }, is_default: false }];
const ignore = () => {};
const clients: QueryClient[] = [];

function artifact(id: string, mediaType = "image/png"): Artifact {
  return { id, sha256: "a".repeat(64), kind: "input", media_type: mediaType, size_bytes: 10,
    original_name: `${id}.png`, metadata_json: {}, created_at: stamp };
}

function editorState(initial: Partial<TurnEditorState> = {}): TurnEditorState {
  return { requestId: "request-initial", mode: "image", attachments: [], attachmentIntent: "replace",
    mentions: [], referenceIntent: "replace", outputCount: 1, templateSettings: null, ...initial };
}

type HarnessProps = {
  revisionId?: string;
  settingsRole?: EngineRole;
  normalWorkflow?: boolean;
  initial?: Partial<TurnEditorState>;
  initialText?: string;
  onAccept?: (submission: TurnEditorSubmission) => Promise<unknown>;
  onSend?: TurnEditorProps["onSend"];
  onStopAndSend?: TurnEditorProps["onStopAndSend"];
  stoppable?: boolean;
  remountable?: boolean;
  controlled?: boolean;
  classificationSource?: TurnEditorProps["classificationSource"];
  sourceFitPreviewContext?: TurnEditorProps["sourceFitPreviewContext"];
};
function Harness({ initial, initialText = "Paint a green landscape", onAccept, onSend = ignore,
  onStopAndSend = ignore, stoppable = false, remountable = false, controlled = true, classificationSource, revisionId, settingsRole = "image", normalWorkflow = false, sourceFitPreviewContext }: HarnessProps) {
  const [draft, setDraft] = useState<ComposerDraft>({ text: initialText, promptSource: null });
  const [state, setState] = useState(() => editorState(initial));
  const [settings, setSettings] = useState<Record<string, unknown>>({ width: 640 });
  const [presetId, setPresetId] = useState<string | null>(null);
  const [visible, setVisible] = useState(true);
  return <>
    {remountable && <button onClick={() => setVisible((current) => !current)}>Toggle editor</button>}
    {visible && <TurnEditor chat={chat} engines={[engine]}
      stoppable={stoppable} settings={settings} onSettings={setSettings} settingsRole={settingsRole} onSettingsRole={ignore}
      presetId={presetId} onPreset={setPresetId} onMode={ignore} onSend={onSend} onStop={ignore} onStopAndSend={onStopAndSend}
      sourceFitPreviewContext={sourceFitPreviewContext}
      maxMediaOutputsPerPlan={4} draft={draft} onDraftChange={setDraft}
      {...(controlled ? { editorState: state, onEditorStateChange: setState } : { initialState: editorState(initial) })}
      classificationSource={classificationSource} contextMessages={[]} contextVisualArtifacts={classificationSource ? [artifact("prior-image")] : []}
      workflowSelection={revisionId ? { selector_capability: "image", mode: "revision",
        workflow_revision_id: revisionId, workflow_family_id: null, legacy_profile_id: null } : undefined}
      workflowControl={normalWorkflow ? undefined : <span>Draft workflow</span>} workflowSchemaOverride={null} onAccept={onAccept}
      submitLabel={onAccept ? "Queue edited version" : "Send"} />}
  </>;
}

function mount(element: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  return render(<QueryClientProvider client={client}>{element}</QueryClientProvider>);
}

beforeEach(() => {
  vi.mocked(api.profilesPage).mockResolvedValue([]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.workflowRevisionSourceFit).mockResolvedValue({
    available: false, reason: "source_fit_workflow_unsupported", modes: [], request_authorized: false,
  });
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.upload).mockImplementation(async (file) => artifact(file.name, file.type));
});
afterEach(() => {
  cleanup();
  for (const client of clients.splice(0)) client.clear();
  vi.resetAllMocks();
});

describe("TurnEditor", () => {
  it("initializes a local draft from the source mode and count rather than current chat defaults", async () => {
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockRejectedValue(new Error("Queue full"));
    mount(<Harness controlled={false} onAccept={accept} initial={{ mode: "video", outputCount: 3, requestId: "source-request" }} />);
    expect(screen.getByLabelText("Generation mode")).toHaveValue("video");
    expect(screen.getByLabelText("Number of outputs")).toHaveValue("3");
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    await screen.findByRole("alert");
    expect(accept.mock.calls[0][0]).toMatchObject({ requestId: "source-request", mode: "video", outputCount: 3 });
  });

  it("submits an explicit single output when reducing an initialized multi-output draft", async () => {
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockResolvedValue({});
    mount(<Harness onAccept={accept} initial={{ outputCount: 3 }} />);
    expect(screen.getByLabelText("Number of outputs")).toHaveValue("3");
    fireEvent.change(screen.getByLabelText("Number of outputs"), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(1));
    expect(accept.mock.calls[0][0]).toMatchObject({ mode: "image", outputCount: 1 });
  });

  it("preserves the normal composer's attachments, references, count and stop-and-send dispatch", () => {
    const send = vi.fn();
    mount(<Harness onStopAndSend={send} stoppable initialText="  Paint @ada near the lake  " initial={{
      outputCount: 3, attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }],
      mentions: [{ referenceSubjectId: "subject-one", mentionSlug: "ada" }],
    }} />);
    fireEvent.click(screen.getByRole("button", { name: "Stop current response and send" }));
    expect(send).toHaveBeenCalledWith("Paint @ada near the lake", "image", ["source-image"], { width: 640 },
      [{ reference_subject_id: "subject-one", source: "mention" }], 3, undefined);
    expect(screen.getByLabelText("Message")).toHaveValue("");
    expect(screen.queryByLabelText("Preview source-image")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Number of outputs")).toHaveValue("1");
  });

  it("retains the exact draft and request identity after rejection, including across a remount", async () => {
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>()
      .mockRejectedValueOnce(new Error("Queue full"))
      .mockResolvedValueOnce({});
    mount(<Harness onAccept={accept} remountable initialText={"  Paint @ada near the lake\n"} initial={{
      outputCount: 2, attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }],
      mentions: [{ referenceSubjectId: "subject-one", mentionSlug: "ada" }],
    }} />);
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Queue full");
    expect(screen.getByLabelText("Message")).toHaveValue("  Paint @ada near the lake\n");
    expect(screen.getByLabelText("Preview source-image")).toBeInTheDocument();
    expect(screen.getByLabelText("Number of outputs")).toHaveValue("2");
    fireEvent.click(screen.getByRole("button", { name: "Toggle editor" }));
    fireEvent.click(screen.getByRole("button", { name: "Toggle editor" }));
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    await waitFor(() => expect(screen.getByLabelText("Message")).toHaveValue(""));
    expect(accept).toHaveBeenCalledTimes(2);
    expect(accept.mock.calls[1][0]).toEqual(accept.mock.calls[0][0]);
    expect(accept.mock.calls[0][0]).toMatchObject({ requestId: "request-initial", inputArtifactIds: ["source-image"],
      references: [{ reference_subject_id: "subject-one", source: "mention" }], outputCount: 2, settings: { width: 640 } });
  });

  it("retains inherited bindings and sends explicit empty lists when they are removed", async () => {
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockRejectedValue(new Error("Try later"));
    mount(<Harness onAccept={accept} initialText="Paint @ada near the lake" initial={{
      attachmentIntent: "inherit", referenceIntent: "inherit",
      attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }],
      mentions: [{ referenceSubjectId: "subject-one", mentionSlug: "ada" }],
    }} />);
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    await screen.findByRole("alert");
    expect(accept.mock.calls[0][0].inputArtifactIds).toBeUndefined();
    expect(accept.mock.calls[0][0].references).toBeUndefined();
    fireEvent.click(screen.getByRole("button", { name: /^Remove Uploaded image:/ }));
    fireEvent.change(screen.getByLabelText("Message"), { target: { value: "Paint a lake" } });
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(2));
    expect(accept.mock.calls[1][0]).toMatchObject({ inputArtifactIds: [], references: [] });
    expect(accept.mock.calls[1][0].requestId).not.toBe(accept.mock.calls[0][0].requestId);
  });

  it("replaces inherited inputs with the ordered library and uploaded selections", async () => {
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockResolvedValue({});
    const { container } = mount(<Harness onAccept={accept} initial={{ attachmentIntent: "inherit",
      attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }],
    }} />);
    fireEvent.click(screen.getByRole("button", { name: /^Remove Uploaded image:/ }));
    fireEvent.click(screen.getByRole("button", { name: "Attach from the library" }));
    fireEvent.click(screen.getByRole("button", { name: "Choose library image" }));
    const file = new File(["pixels"], "second", { type: "image/png" });
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [file] } });
    await screen.findByLabelText("Preview second.png");
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(1));
    expect(accept.mock.calls[0][0].inputArtifactIds).toEqual(["library-image", "second"]);
    expect(api.upload).toHaveBeenCalledTimes(1);
  });

  it("blocks repeat submission while acceptance is pending and keeps the draft visible", async () => {
    let finish: (result: unknown) => void = ignore;
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    mount(<Harness onAccept={accept} stoppable />);
    const field = screen.getByLabelText("Message");
    fireEvent.keyDown(field, { key: "Enter" });
    fireEvent.keyDown(field, { key: "Enter" });
    expect(accept).toHaveBeenCalledTimes(1);
    expect(field).toHaveValue("Paint a green landscape");
    expect(screen.getByRole("button", { name: "Queue edited version" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Stop current response" })).not.toBeInTheDocument();
    await act(async () => finish({}));
    expect(field).toHaveValue("");
  });

  it("keeps drag/drop and clipboard uploads ordered and blocks sending an unfinished upload", async () => {
    let finishUpload: (value: Artifact) => void = ignore;
    vi.mocked(api.upload).mockImplementationOnce(() => new Promise((resolve) => { finishUpload = resolve; }));
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockResolvedValue({});
    const { container } = mount(<Harness onAccept={accept} />);
    const dropped = new File(["image"], "dropped", { type: "image/png" });
    const field = screen.getByLabelText("Message");
    fireEvent.drop(container.querySelector(".composer-wrap")!, { dataTransfer: { files: [dropped], types: ["Files"] } });
    fireEvent.keyDown(field, { key: "Enter" });
    expect(accept).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Queue edited version" })).toBeDisabled();
    await act(async () => finishUpload(artifact("dropped")));
    const pasted = new File(["image"], "pasted", { type: "image/png" });
    fireEvent.paste(field, { clipboardData: { files: [pasted] } });
    await screen.findByLabelText("Preview pasted.png");
    fireEvent.keyDown(field, { key: "Enter" });
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(1));
    expect(accept.mock.calls[0][0].inputArtifactIds).toEqual(["dropped", "pasted"]);
  });

  it("uses shared settings and count controls with draft-local values and preset identity", async () => {
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockResolvedValue({});
    mount(<Harness onAccept={accept} />);
    fireEvent.change(screen.getByLabelText("Number of outputs"), { target: { value: "4" } });
    fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
    fireEvent.change(await screen.findByRole("spinbutton", { name: "Width" }), { target: { value: "1024" } });
    await screen.findByRole("option", { name: "Landscape" });
    fireEvent.change(screen.getByLabelText("image preset"), { target: { value: "preset-one" } });
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(1));
    expect(accept.mock.calls[0][0]).toMatchObject({ settings: { width: 1024 }, outputCount: 4, presetId: "preset-one" });
    expect(api.setChatWorkflowSelection).not.toHaveBeenCalled();
  });
});
it("classifies inherited source media using the edited source identity", async () => {
  vi.mocked(api.classifyDraft).mockResolvedValue({ references_prior_visual: true });
  const binding = { source_message_id: "source-user", source_run_id: "source-run", source_snapshot_sha256: "a".repeat(64) };
  mount(<Harness classificationSource={binding} onAccept={async () => {}} />);
  await waitFor(() => expect(api.classifyDraft).toHaveBeenCalledWith(
    chat.id, "Paint a green landscape", "image", binding,
  ));
  expect(await screen.findByRole("button", { name: "Open editing studio" })).toBeEnabled();
});

describe("source canvas in the composer", () => {
  function preview(sourceId = "source-image", revisionId = "revision-canvas") {
    return {
      version: 1 as const, mode: "extend" as const,
      workflow_revision_id: revisionId, workflow_artifact_sha256: "a".repeat(64),
      source_artifact_id: sourceId,
      source: { width: 400, height: 300 }, canvas: { width: 1200, height: 900 },
      margins: { left: 400, top: 300, right: 400, bottom: 300 },
      source_rectangle: { x: 400, y: 300, width: 400, height: 300 },
      request_authorized: false as const,
    };
  }

  async function chooseCanvas() {
    fireEvent.click(await screen.findByRole("button", { name: "Extend / preserve all" }));
    fireEvent.change(screen.getByLabelText("Canvas width"), { target: { value: "1200" } });
    fireEvent.change(screen.getByLabelText("Canvas height"), { target: { value: "900" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview canvas" }));
  }

  beforeEach(() => {
    vi.mocked(api.workflowRevisionSourceFit).mockResolvedValue({
      available: true, reason: null, modes: ["extend"], request_authorized: false,
    });
    vi.mocked(api.previewWorkflowRevisionSourceFit).mockResolvedValue(preview());
    vi.mocked(api.previewTurnSourceFit).mockResolvedValue(preview());
  });

  it.each([false, true])("previews the complete source then dispatches the exact canvas with stop=%s", async (stop) => {
    const send = vi.fn();
    mount(<Harness revisionId="revision-canvas" stoppable={stop} onSend={send} onStopAndSend={send}
      initial={{ attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }] }} />);
    await chooseCanvas();
    expect(await screen.findByRole("img", { name: /Extension preview/ })).toBeInTheDocument();
    expect(screen.getByText("Source: 400 × 300 · Output: 1200 × 900")).toBeInTheDocument();
    // The preview is fitted within the send itself: the same text, mode, inputs and settings.
    expect(api.previewTurnSourceFit).toHaveBeenLastCalledWith(chat.id, expect.objectContaining({
      text: "Paint a green landscape", mode: "image", input_artifact_ids: ["source-image"], settings: {}, references: [],
      source_fit: { mode: "extend", width: 1200, height: 900 },
    }), expect.any(AbortSignal));
    expect(api.previewWorkflowRevisionSourceFit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: stop ? "Stop current response and send" : "Send" }));
    expect(send).toHaveBeenCalledWith("Paint a green landscape", "image", ["source-image"], {}, [], undefined, undefined, {
      sourceArtifactId: "source-image", workflowRevisionId: "revision-canvas",
      request: { mode: "extend", width: 1200, height: 900 },
    });
    expect(screen.getByLabelText("Message")).toHaveValue("");
    expect(screen.queryByLabelText("Canvas width")).not.toBeInTheDocument();
  });

  it("loads the image workflow for an Auto canvas while text settings are visible", async () => {
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([{
      selector_capability: "image", mode: "revision", workflow_revision_id: "revision-canvas",
      workflow_family_id: null, legacy_profile_id: null,
    }]);
    const send = vi.fn();
    mount(<Harness normalWorkflow settingsRole="chat" onSend={send} initial={{
      mode: "auto", attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }],
    }} />);
    await chooseCanvas();
    await screen.findByRole("img", { name: /Extension preview/ });
    expect(api.chatWorkflowSelections).toHaveBeenCalledWith(chat.id);
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(send).toHaveBeenCalledWith("Paint a green landscape", "auto", ["source-image"], {}, [], undefined, undefined, {
      sourceArtifactId: "source-image", workflowRevisionId: "revision-canvas",
      request: { mode: "extend", width: 1200, height: 900 },
    });
  });

  it("keeps the draft when Send is pressed before the requested preview returns", async () => {
    let complete!: (value: ReturnType<typeof preview>) => void;
    vi.mocked(api.previewTurnSourceFit).mockReturnValue(new Promise((resolve) => { complete = resolve; }));
    const send = vi.fn();
    mount(<Harness revisionId="revision-canvas" onSend={send}
      initial={{ attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }] }} />);
    await chooseCanvas();
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(send).not.toHaveBeenCalled();
    expect(await screen.findByRole("alert")).toHaveTextContent("Preview the selected source canvas before sending.");
    expect(screen.getByLabelText("Message")).toHaveValue("Paint a green landscape");
    await act(async () => { complete(preview()); });
    await screen.findByRole("img", { name: /Extension preview/ });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(send).toHaveBeenCalledTimes(1);
  });

  it("does not treat a response for another source as the requested preview", async () => {
    vi.mocked(api.previewTurnSourceFit).mockResolvedValue(preview("another-image"));
    const send = vi.fn();
    mount(<Harness revisionId="revision-canvas" onSend={send}
      initial={{ attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }] }} />);
    await chooseCanvas();
    await waitFor(() => expect(api.previewTurnSourceFit).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(send).not.toHaveBeenCalled();
    expect(screen.queryByRole("img", { name: /Extension preview/ })).not.toBeInTheDocument();
  });

  it("sends the workflow the server previewed the canvas with", async () => {
    vi.mocked(api.previewTurnSourceFit).mockResolvedValue(preview("source-image", "revision-resolved"));
    const send = vi.fn();
    mount(<Harness revisionId="revision-canvas" onSend={send}
      initial={{ attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }] }} />);
    await chooseCanvas();
    await screen.findByRole("img", { name: /Extension preview/ });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(send.mock.calls[0][2]).toEqual(["source-image"]);
    expect(send.mock.calls[0][7]).toMatchObject({ sourceArtifactId: "source-image", workflowRevisionId: "revision-resolved" });
  });

  it("leaves the workflow's answer to the server and sends no canvas it would not draw", async () => {
    vi.mocked(api.previewTurnSourceFit).mockRejectedValue(new Error("source_fit_workflow_unsupported"));
    const send = vi.fn();
    mount(<Harness revisionId="revision-canvas" onSend={send}
      initial={{ attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }] }} />);
    await chooseCanvas();
    expect(await screen.findByText("This source and workflow cannot use that canvas. Change the dimensions or choose another workflow."))
      .toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(send).not.toHaveBeenCalled();
    expect(api.workflowRevisionSourceFit).not.toHaveBeenCalled();
  });

  it("carries an exact preview into a prior-turn submission and retains it after rejection", async () => {
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockRejectedValue(new Error("Queue full"));
    mount(<Harness revisionId="revision-canvas" onAccept={accept} initial={{
      attachmentIntent: "inherit", attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }],
    }} />);
    await chooseCanvas();
    await screen.findByRole("img", { name: /Extension preview/ });
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Queue full");
    expect(accept.mock.calls[0][0]).toMatchObject({
      inputArtifactIds: ["source-image"], settings: {},
      sourceFit: { sourceArtifactId: "source-image", workflowRevisionId: "revision-canvas",
        request: { mode: "extend", width: 1200, height: 900 } },
    });
    expect(screen.getByLabelText("Canvas width")).toHaveValue(1200);
  });
  it("puts the picture the server fitted the canvas to first among an edit's inputs", async () => {
    vi.mocked(api.previewPriorTurnSourceFit).mockResolvedValue(preview("resolved-image"));
    const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockResolvedValue(undefined);
    mount(<Harness revisionId="revision-canvas" onAccept={accept}
      sourceFitPreviewContext={{ kind: "prior-edit", id: "message-edited", request: { text: "Paint a green landscape", idempotency_key: "edit" } }}
      initial={{ sourceFit: { sourceArtifactId: "", workflowRevisionId: "", request: { mode: "extend", width: 1200, height: 900 } } }} />);
    fireEvent.click(await screen.findByRole("button", { name: "Preview canvas" }));
    const shown = await screen.findByRole("img", { name: /Extension preview/ });
    // The preview draws the picture the server fitted, not whatever the composer holds.
    expect(shown.querySelector("image")).toHaveAttribute("href", "/api/artifacts/resolved-image/content");
    fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(1));
    expect(accept.mock.calls[0][0]).toMatchObject({
      inputArtifactIds: ["resolved-image"], sourceFit: { sourceArtifactId: "resolved-image", workflowRevisionId: "revision-canvas" },
    });
  });
  it("allows a temporarily empty dimension without saving zero or sending the old preview", async () => {
    const send = vi.fn();
    mount(<Harness revisionId="revision-canvas" onSend={send}
      initial={{ attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }] }} />);
    await chooseCanvas();
    await screen.findByRole("img", { name: /Extension preview/ });
    const width = screen.getByLabelText("Canvas width");
    fireEvent.change(width, { target: { value: "" } });
    expect(width).toHaveValue(null);
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(send).not.toHaveBeenCalled();
    fireEvent.blur(width);
    expect(width).toHaveValue(1200);
    expect(screen.queryByRole("img", { name: /Extension preview/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Preview canvas" }));
    await screen.findByRole("img", { name: /Extension preview/ });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(send.mock.calls[0][7].request).toEqual({ mode: "extend", width: 1200, height: 900 });
  });

  it("covers an attached picture and its file name while pictures are covered", () => {
    localStorage.setItem(SENSITIVE_MEDIA_KEY, "hide");
    try {
      mount(<Harness initial={{ attachments: [{ id: "source-image", kind: "image", origin: "uploaded" }] }} />);

      const preview = screen.getByLabelText("Preview Uploaded image");
      expect(preview.querySelector("img")).toBeNull();
      expect(screen.queryByText("source-image")).toBeNull();
    } finally {
      localStorage.removeItem(SENSITIVE_MEDIA_KEY);
    }
  });

});
