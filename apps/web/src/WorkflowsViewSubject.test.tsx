import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import { openWorkflowEditorPopup, runWorkflowEditor } from "./workflowEditorBridge";

vi.mock("./api", () => ({
  api: {
    workflows: vi.fn(),
    workflowFamilies: vi.fn().mockResolvedValue([]),
    validateWorkflow: vi.fn(),
    cloneWorkflow: vi.fn(),
    updateWorkflow: vi.fn(),
    createWorkflow: vi.fn(),
    createWorkflowRevision: vi.fn(),
    createWorkflowEditorDraft: vi.fn(),
    consumeWorkflowEditor: vi.fn(),
    cancelWorkflowEditor: vi.fn().mockResolvedValue(undefined),
    workflowOpenTarget: vi.fn(),
    registryInstalls: vi.fn().mockResolvedValue([]),
    customNodes: vi.fn().mockResolvedValue([]),
  },
}));

vi.mock("./workflowEditorBridge", () => ({
  openWorkflowEditorPopup: vi.fn(),
  runWorkflowEditor: vi.fn(),
}));

function revision(id: string) {
  return {
    id,
    workflow_id: id,
    version: 1,
    engine: "comfyui",
    api_graph_json: {},
    ui_graph_json: {},
    input_schema_json: {
      required: ["steps"],
      properties: {
        steps: { type: "integer", default: 20, minimum: 1, maximum: 50, title: "Steps" },
        sampler: { type: "string", enum: ["euler", "dpmpp"], default: "euler" },
      },
    },
    dependencies_json: {},
    trusted: true,
    created_at: "2026-08-06T00:00:00Z",
  };
}

function workflow(id: string, name: string) {
  return {
    id,
    name,
    description: "",
    operation: "text_to_image",
    current_revision_id: id,
    revisions: [revision(id)],
  };
}

function editorReturn() {
  return {
    validated_return_id: "return-1",
    session_id: "session-1",
    workflow_id: "wf-a",
    base_revision_id: "revision-1",
    current_revision_id: "revision-1",
    base_graph_sha256: "a".repeat(64),
    returned_graph_sha256: "b".repeat(64),
    base_prompt_sha256: "c".repeat(64),
    returned_prompt_sha256: "d".repeat(64),
    changed: true,
    forked: false,
    delta: {
      node_count_delta: 0,
      link_count_delta: 0,
      added_node_types: [],
      removed_node_types: [],
      added_asset_filenames: [],
      removed_asset_filenames: [],
    },
    expires_at: "2026-08-08T18:00:00Z",
  };
}

function renderView() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <WorkflowsView />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("workflow creation requests", () => {
  it.each([
    { action: "New workflow", submit: "Save workflow", newRevision: false },
    { action: "New revision", submit: "Create revision", newRevision: true },
  ])("omits browser trust when saving $action", async ({ action, submit, newRevision }) => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    vi.mocked(api.createWorkflow).mockResolvedValue(workflow("wf-new", "Created") as never);
    vi.mocked(api.createWorkflowRevision).mockResolvedValue(revision("revision-new") as never);

    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    expect(screen.getByText("Trusted", { selector: ".badge" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: action }));

    expect.soft(screen.queryByRole("checkbox", { name: /trust this workflow/i })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: submit }));

    const save = newRevision ? api.createWorkflowRevision : api.createWorkflow;
    await waitFor(() => expect(save).toHaveBeenCalledOnce());
    const payload = newRevision
      ? vi.mocked(api.createWorkflowRevision).mock.calls[0]?.[1]
      : vi.mocked(api.createWorkflow).mock.calls[0]?.[0];
    expect(payload).toEqual(expect.objectContaining({ api_graph: {}, ui_graph: {} }));
    expect(payload).not.toHaveProperty("trusted");
  });
});

describe("a verdict belongs to the workflow it was asked about", () => {
  it("does not carry one workflow's validation over to another", async () => {
    vi.mocked(api.workflows).mockResolvedValue([
      workflow("wf-a", "Alpha"),
      workflow("wf-b", "Beta"),
    ] as never);
    vi.mocked(api.validateWorkflow).mockResolvedValue({
      valid: false,
      errors: ["Alpha is missing a node"],
      warnings: [],
    } as never);

    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(await screen.findByRole("button", { name: "Validate" }));
    await waitFor(() => expect(screen.getByText("Alpha is missing a node")).toBeTruthy());

    // The verdict was about Alpha. Beta has not been validated at all.
    fireEvent.click(screen.getByText("Beta"));

    expect(screen.queryByText("Alpha is missing a node")).toBeNull();
  });
});

describe("a list that could not be read", () => {
  it("says so instead of offering an empty shelf", async () => {
    vi.mocked(api.workflows).mockRejectedValue(new Error("workflow library unreachable"));

    renderView();

    await waitFor(() =>
      expect(screen.getByText("workflow library unreachable")).toBeTruthy(),
    );
  });
});

describe("opening a workflow in ComfyUI", () => {
  it("opens the shell during the click before beginning asynchronous setup", async () => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    const popup = { closed: false, close: vi.fn(), postMessage: vi.fn() } as unknown as Window;
    vi.mocked(openWorkflowEditorPopup).mockReturnValue(popup);
    let finish!: (value: never) => void;
    vi.mocked(runWorkflowEditor).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));

    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "Edit in ComfyUI (preview)" }));

    expect(openWorkflowEditorPopup).toHaveBeenCalledOnce();
    await waitFor(() => expect(runWorkflowEditor).toHaveBeenCalledWith(
      api,
      "wf-a",
      popup,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    ));
    finish({ kind: "unchanged", returned: { changed: false } } as never);
    await screen.findByText("No workflow changes to save.");
  });

  it("opens only one session across a rapid double click", async () => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    const popup = { closed: false, close: vi.fn(), postMessage: vi.fn() } as unknown as Window;
    vi.mocked(openWorkflowEditorPopup).mockReturnValue(popup);
    vi.mocked(runWorkflowEditor).mockImplementation(() => new Promise(() => {}));
    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    const edit = screen.getByRole("button", { name: "Edit in ComfyUI (preview)" });
    fireEvent.click(edit);
    fireEvent.click(edit);
    expect(openWorkflowEditorPopup).toHaveBeenCalledOnce();
    await waitFor(() => expect(runWorkflowEditor).toHaveBeenCalledOnce());
  });

  it("keeps an ambiguous returned edit until retry or explicit discard", async () => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    const popup = { closed: false, close: vi.fn(), postMessage: vi.fn() } as unknown as Window;
    vi.mocked(openWorkflowEditorPopup).mockReturnValue(popup);
    vi.mocked(runWorkflowEditor).mockImplementation(async (_client, _id, _popup, options) => {
      options?.onSubmission?.({
        workflowId: "wf-a",
        sessionId: "session-1",
        nonce: "nonce-1",
        baseRevisionId: "revision-1",
        uiGraph: { nodes: [1] },
        apiPrompt: { 1: {} },
      });
      throw new Error("response was lost");
    });
    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "Edit in ComfyUI (preview)" }));
    expect(await screen.findByRole("button", { name: "Retry validating returned edit" })).toBeTruthy();

    const disabledEdit = screen.getByRole("button", { name: "Edit in ComfyUI (preview)" });
    expect((disabledEdit as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(disabledEdit);
    expect(openWorkflowEditorPopup).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole("button", { name: "Discard pending edit" }));
    await waitFor(() => expect(api.cancelWorkflowEditor).toHaveBeenCalledWith(
      "wf-a",
      "session-1",
      "nonce-1",
    ));
    expect(screen.queryByRole("button", { name: "Retry validating returned edit" })).toBeNull();
    expect(screen.getByText("Pending workflow edit discarded.")).toBeTruthy();
  });

  it("does not offer recovery controls while the original consume is running", async () => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    const popup = { closed: false, close: vi.fn(), postMessage: vi.fn() } as unknown as Window;
    vi.mocked(openWorkflowEditorPopup).mockReturnValue(popup);
    let fail!: (reason: Error) => void;
    vi.mocked(runWorkflowEditor).mockImplementation((_client, _id, _popup, options) => {
      options?.onSubmission?.({
        workflowId: "wf-a",
        sessionId: "session-1",
        nonce: "nonce-1",
        baseRevisionId: "revision-1",
        uiGraph: { nodes: [1] },
        apiPrompt: { 1: {} },
      });
      return new Promise((_resolve, reject) => { fail = reject; });
    });
    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "Edit in ComfyUI (preview)" }));
    await waitFor(() => expect(runWorkflowEditor).toHaveBeenCalledOnce());
    expect(screen.queryByRole("button", { name: "Retry validating returned edit" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Discard pending edit" })).toBeNull();

    fail(new Error("response was lost"));
    expect(await screen.findByRole("button", { name: "Retry validating returned edit" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Discard pending edit" })).toBeTruthy();
  });

  it("does not offer recovery controls while the original draft save is running", async () => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    const popup = { closed: false, close: vi.fn(), postMessage: vi.fn() } as unknown as Window;
    vi.mocked(openWorkflowEditorPopup).mockReturnValue(popup);
    let fail!: (reason: Error) => void;
    vi.mocked(runWorkflowEditor).mockImplementation((_client, _id, _popup, options) => {
      options?.onValidated?.(editorReturn());
      return new Promise((_resolve, reject) => { fail = reject; });
    });
    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "Edit in ComfyUI (preview)" }));
    await waitFor(() => expect(runWorkflowEditor).toHaveBeenCalledOnce());
    expect(screen.queryByRole("button", { name: "Retry saving validated edit" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Discard pending edit" })).toBeNull();

    fail(new Error("draft save failed"));
    expect(await screen.findByRole("button", { name: "Retry saving validated edit" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Discard pending edit" })).toBeTruthy();
  });

  it("does not replace a validated edit whose draft still needs saving", async () => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    const popup = { closed: false, close: vi.fn(), postMessage: vi.fn() } as unknown as Window;
    vi.mocked(openWorkflowEditorPopup).mockReturnValue(popup);
    vi.mocked(runWorkflowEditor).mockImplementation(async (_client, _id, _popup, options) => {
      options?.onValidated?.(editorReturn());
      throw new Error("draft save failed");
    });
    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "Edit in ComfyUI (preview)" }));
    expect(await screen.findByRole("button", { name: "Retry saving validated edit" })).toBeTruthy();

    const edit = screen.getByRole("button", { name: "Edit in ComfyUI (preview)" });
    expect((edit as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(edit);
    expect(openWorkflowEditorPopup).toHaveBeenCalledOnce();
  });

  it("explains when the browser refuses the synchronous popup", async () => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    vi.mocked(openWorkflowEditorPopup).mockReturnValue(null);

    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "Edit in ComfyUI (preview)" }));

    expect(await screen.findByText(/browser blocked the workflow editor window/i)).toBeTruthy();
    expect(runWorkflowEditor).not.toHaveBeenCalled();
  });

  it("keeps the manual graph handoff available until browser certification", async () => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    vi.mocked(api.workflowOpenTarget).mockResolvedValue({
      url: "http://127.0.0.1:8188/",
      filename: "alpha.json",
      ui_graph: {},
    } as never);
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "Download UI graph" }));

    const link = await screen.findByRole("link", { name: "Open ComfyUI manually" });
    expect(link.getAttribute("href")).toBe("http://127.0.0.1:8188/");
  });
});

describe("after a workflow changes", () => {
  it("re-asks for the families the change could have moved", async () => {
    // The server derives a family's current revision, engine, capabilities and
    // readiness from the revision that just changed. Refreshing the list alone
    // left the families beside it - and the selectors elsewhere - answering
    // from before the change, with nothing to heal them while mounted.
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    vi.mocked(api.cloneWorkflow).mockResolvedValue(workflow("wf-b", "Alpha copy") as never);

    renderView();
    await screen.findByText("Alpha");
    const familiesReadBefore = vi.mocked(api.workflowFamilies).mock.calls.length;

    fireEvent.click(screen.getByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "Duplicate" }));

    await waitFor(() =>
      expect(vi.mocked(api.workflowFamilies).mock.calls.length).toBeGreaterThan(familiesReadBefore),
    );
  });
});

describe("saving a new revision", () => {
  async function openTheEditor() {
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "New revision" }));
  }

  it("does not rewrite the name when only the graph changed", async () => {
    // The metadata write is unvalidated and commits first, so a revision the
    // server rejects used to leave a rename behind that nobody asked for.
    // With nothing to rename there is no second write to be left behind.
    vi.mocked(api.createWorkflowRevision).mockRejectedValue(new Error("schema rejected"));
    await openTheEditor();

    fireEvent.click(screen.getByRole("button", { name: "Create revision" }));

    await waitFor(() => expect(screen.getByText("schema rejected")).toBeTruthy());
    expect(api.updateWorkflow).not.toHaveBeenCalled();
  });

  it("shows a rename that landed even though the revision was refused", async () => {
    vi.mocked(api.updateWorkflow).mockResolvedValue({} as never);
    vi.mocked(api.createWorkflowRevision).mockRejectedValue(new Error("schema rejected"));
    await openTheEditor();

    const nameBox = screen.getByDisplayValue("Alpha");
    fireEvent.change(nameBox, { target: { value: "Alpha renamed" } });
    const readsBefore = vi.mocked(api.workflows).mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Create revision" }));

    await waitFor(() => expect(api.updateWorkflow).toHaveBeenCalled());
    // The rename is committed, so the list has to be re-read despite the
    // failure - otherwise the old name stays on screen until some later visit.
    await waitFor(() =>
      expect(vi.mocked(api.workflows).mock.calls.length).toBeGreaterThan(readsBefore),
    );
  });
});

describe("the controls a revision declares", () => {
  it("describes them rather than pretending they can be set here", async () => {
    // They were live inputs with local state and no consumer: typing changed
    // a value read by nobody and discarded when the selection moved. A field
    // that throws away what you put in it is worse than plain text.
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);

    renderView();
    fireEvent.click(await screen.findByText("Alpha"));

    expect(screen.getByText("Steps")).toBeTruthy();
    expect(screen.getByText("Default: 20")).toBeTruthy();
    expect(screen.getByText("One of: euler, dpmpp")).toBeTruthy();
    expect(screen.queryByRole("spinbutton")).toBeNull();
    expect(screen.queryByRole("combobox", { name: /sampler/i })).toBeNull();
  });
});
