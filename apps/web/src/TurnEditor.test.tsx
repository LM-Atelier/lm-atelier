import { useState, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import type { ComposerDraft } from "./composerPromptSource";
import { TurnEditor, type TurnEditorProps, type TurnEditorState, type TurnEditorSubmission } from "./TurnEditor";
import type { Artifact, ChatDetail, EngineCapabilities, GenerationPreset } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api,
    workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(), projectWorkflowSelections: vi.fn(),
    classifyDraft: vi.fn(), references: vi.fn(), upload: vi.fn(), setChatWorkflowSelection: vi.fn(),
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
  initial?: Partial<TurnEditorState>;
  initialText?: string;
  onAccept?: (submission: TurnEditorSubmission) => Promise<unknown>;
  onSend?: TurnEditorProps["onSend"];
  onStopAndSend?: TurnEditorProps["onStopAndSend"];
  stoppable?: boolean;
  remountable?: boolean;
  controlled?: boolean;
  classificationSource?: TurnEditorProps["classificationSource"];
};
function Harness({ initial, initialText = "Paint a green landscape", onAccept, onSend = ignore,
  onStopAndSend = ignore, stoppable = false, remountable = false, controlled = true, classificationSource }: HarnessProps) {
  const [draft, setDraft] = useState<ComposerDraft>({ text: initialText, promptSource: null });
  const [state, setState] = useState(() => editorState(initial));
  const [settings, setSettings] = useState<Record<string, unknown>>({ width: 640 });
  const [presetId, setPresetId] = useState<string | null>(null);
  const [visible, setVisible] = useState(true);
  return <>
    {remountable && <button onClick={() => setVisible((current) => !current)}>Toggle editor</button>}
    {visible && <TurnEditor chat={chat} engines={[engine]} profiles={[]} workflows={[]} presets={presets}
      stoppable={stoppable} settings={settings} onSettings={setSettings} settingsRole="image" onSettingsRole={ignore}
      presetId={presetId} onPreset={setPresetId} onMode={ignore} onSend={onSend} onStop={ignore} onStopAndSend={onStopAndSend}
      maxMediaOutputsPerPlan={4} draft={draft} onDraftChange={setDraft}
      {...(controlled ? { editorState: state, onEditorStateChange: setState } : { initialState: editorState(initial) })}
      classificationSource={classificationSource} contextMessages={[]} contextVisualArtifacts={classificationSource ? [artifact("prior-image")] : []}
      workflowControl={<span>Draft workflow</span>} workflowSchemaOverride={null} onAccept={onAccept}
      submitLabel={onAccept ? "Queue edited version" : "Send"} />}
  </>;
}

function mount(element: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  return render(<QueryClientProvider client={client}>{element}</QueryClientProvider>);
}

beforeEach(() => {
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
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
    fireEvent.change(screen.getByRole("spinbutton", { name: "Width" }), { target: { value: "1024" } });
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
