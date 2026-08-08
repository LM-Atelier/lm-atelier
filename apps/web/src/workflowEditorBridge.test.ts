import { afterEach, describe, expect, it, vi } from "vitest";
import {
  openWorkflowEditorPopup,
  runWorkflowEditor,
  type WorkflowEditorApi,
} from "./workflowEditorBridge";
import type {
  WorkflowEditorDraft,
  WorkflowEditorReturn,
  WorkflowEditorSession,
} from "./types";

const bridgeEnvelope = (type: string, details: Record<string, unknown> = {}) => ({
  source: "lm-atelier-workflow-editor",
  protocol: 2,
  type,
  ...details,
});

class FakePort {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onmessageerror: (() => void) | null = null;
  peer: FakePort | null = null;
  closed = false;

  postMessage(data: unknown) {
    const target = this.peer;
    queueMicrotask(() => target?.onmessage?.({ data } as MessageEvent));
  }

  start() {}

  close() {
    this.closed = true;
  }
}

function fakeChannel() {
  const port1 = new FakePort();
  const port2 = new FakePort();
  port1.peer = port2;
  port2.peer = port1;
  return { port1, port2 };
}

function session(overrides: Partial<WorkflowEditorSession> = {}): WorkflowEditorSession {
  return {
    id: "session-1",
    protocol_version: 2,
    workflow_id: "workflow-1",
    base_revision_id: "revision-1",
    base_graph_sha256: "a".repeat(64),
    base_prompt_sha256: "b".repeat(64),
    created_at: new Date().toISOString(),
    expires_at: new Date(Date.now() + 60_000).toISOString(),
    ui_graph: { nodes: [] },
    nonce: "nonce-1",
    ...overrides,
  };
}

function returned(overrides: Partial<WorkflowEditorReturn> = {}): WorkflowEditorReturn {
  return {
    validated_return_id: "return-1",
    session_id: "session-1",
    workflow_id: "workflow-1",
    base_revision_id: "revision-1",
    current_revision_id: "revision-1",
    base_graph_sha256: "a".repeat(64),
    returned_graph_sha256: "c".repeat(64),
    base_prompt_sha256: "b".repeat(64),
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
    expires_at: new Date(Date.now() + 60_000).toISOString(),
    ...overrides,
  };
}

function draft(): WorkflowEditorDraft {
  return {
    workflow_id: "workflow-1",
    base_revision_id: "revision-1",
    draft_revision_id: "revision-draft",
    current_revision_id: "revision-1",
    version: 2,
    created: true,
    forked: false,
    trusted: false,
    review_required: true,
  };
}

function fakeApi(overrides: Partial<WorkflowEditorApi> = {}): WorkflowEditorApi {
  return {
    startWorkflowEditor: vi.fn().mockResolvedValue(session()),
    consumeWorkflowEditor: vi.fn().mockResolvedValue(returned()),
    createWorkflowEditorDraft: vi.fn().mockResolvedValue(draft()),
    cancelWorkflowEditor: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  };
}

function harness(options: {
  save?: Record<string, unknown>;
  readyOrigin?: string;
  saveNonce?: string;
} = {}) {
  let messageListener: ((event: MessageEvent) => void) | null = null;
  const timers = new Map<number, () => void>();
  const polls = new Map<number, () => void>();
  let nextTimer = 1;
  const channel = fakeChannel();
  const popup = {
    closed: false,
    close: vi.fn(function close() { popup.closed = true; }),
    postMessage: vi.fn((_message: unknown, _origin: string, transfer?: Transferable[]) => {
      const bridgePort = transfer?.[0] as unknown as FakePort;
      bridgePort.onmessage = (event) => {
        const message = event.data as Record<string, unknown>;
        if (message.type !== "load") return;
        bridgePort.postMessage(bridgeEnvelope("loaded", { nonce: message.nonce }));
        if (options.save) {
          bridgePort.postMessage(bridgeEnvelope("save", {
            nonce: options.saveNonce ?? message.nonce,
            graph: options.save.graph,
            prompt: options.save.prompt,
          }));
        }
      };
      bridgePort.postMessage(bridgeEnvelope("connected"));
    }),
  };
  const environment = {
    origin: "http://127.0.0.1:12340",
    addMessageListener(listener: (event: MessageEvent) => void) {
      messageListener = listener;
      queueMicrotask(() => listener(new MessageEvent("message", {
        data: bridgeEnvelope("ready"),
        origin: options.readyOrigin ?? "http://127.0.0.1:12340",
        source: popup as unknown as Window,
      })));
    },
    removeMessageListener(listener: (event: MessageEvent) => void) {
      if (messageListener === listener) messageListener = null;
    },
    createChannel: () => channel as unknown as MessageChannel,
    setTimer(callback: () => void) {
      const id = nextTimer++;
      timers.set(id, callback);
      return id;
    },
    clearTimer(id: number) { timers.delete(id); },
    setPoll(callback: () => void) {
      const id = nextTimer++;
      polls.set(id, callback);
      return id;
    },
    clearPoll(id: number) { polls.delete(id); },
  };
  return {
    popup,
    environment,
    channel,
    fireTimeout: () => [...timers.values()][0]?.(),
    firePoll: () => [...polls.values()].forEach((callback) => callback()),
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("native workflow editor orchestration", () => {
  it("opens the same-origin shell synchronously", () => {
    const popup = {} as Window;
    const open = vi.fn().mockReturnValue(popup);
    vi.stubGlobal("open", open);

    expect(openWorkflowEditorPopup()).toBe(popup);
    expect(open).toHaveBeenCalledWith("/api/workflow-editor/shell", "_blank", "popup");
  });

  it("loads, validates, and stores a changed workflow as a review draft", async () => {
    const validatedReturn = returned();
    const api = fakeApi({
      consumeWorkflowEditor: vi.fn().mockResolvedValue(validatedReturn),
    });
    const browser = harness({ save: { graph: { nodes: [1] }, prompt: { 1: {} } } });
    const phases: string[] = [];
    const validated: Array<WorkflowEditorReturn | null> = [];

    const result = await runWorkflowEditor(
      api,
      "workflow-1",
      browser.popup,
      {
        environment: browser.environment as never,
        onPhase: (phase) => phases.push(phase),
        onValidated: (value) => validated.push(value),
      },
    );

    expect(result).toEqual({ kind: "draft", returned: validatedReturn, draft: draft() });
    expect(api.consumeWorkflowEditor).toHaveBeenCalledWith("workflow-1", "session-1", {
      nonce: "nonce-1",
      base_revision_id: "revision-1",
      ui_graph: { nodes: [1] },
      api_prompt: { 1: {} },
    });
    expect(api.createWorkflowEditorDraft).toHaveBeenCalledWith("workflow-1", "return-1");
    expect(api.cancelWorkflowEditor).not.toHaveBeenCalled();
    expect(phases).toEqual(["preparing", "connecting", "loading", "editing", "validating", "saving"]);
    expect(validated).toEqual([validatedReturn, null]);
    expect(browser.popup.close).toHaveBeenCalledOnce();
  });

  it("does not create a draft for an unchanged return", async () => {
    const unchanged = returned({ changed: false });
    const api = fakeApi({ consumeWorkflowEditor: vi.fn().mockResolvedValue(unchanged) });
    const browser = harness({ save: { graph: { nodes: [] }, prompt: {} } });

    const result = await runWorkflowEditor(
      api,
      "workflow-1",
      browser.popup,
      { environment: browser.environment as never },
    );

    expect(result).toEqual({ kind: "unchanged", returned: unchanged });
    expect(api.createWorkflowEditorDraft).not.toHaveBeenCalled();
  });

  it("cancels the server session when the bridge protocol is incompatible", async () => {
    const api = fakeApi({
      startWorkflowEditor: vi.fn().mockResolvedValue(session({ protocol_version: 99 })),
    });
    const browser = harness();

    await expect(runWorkflowEditor(
      api,
      "workflow-1",
      browser.popup,
      { environment: browser.environment as never },
    )).rejects.toThrow("bridge version is not supported");

    await vi.waitFor(() => expect(api.cancelWorkflowEditor).toHaveBeenCalledWith(
      "workflow-1",
      "session-1",
      "nonce-1",
    ));
  });

  it("retains the validated return when creating its draft fails", async () => {
    const validatedReturn = returned();
    const api = fakeApi({
      consumeWorkflowEditor: vi.fn().mockResolvedValue(validatedReturn),
      createWorkflowEditorDraft: vi.fn().mockRejectedValue(new Error("review dependencies")),
    });
    const browser = harness({ save: { graph: { nodes: [1] }, prompt: { 1: {} } } });
    const validated: Array<WorkflowEditorReturn | null> = [];

    await expect(runWorkflowEditor(
      api,
      "workflow-1",
      browser.popup,
      {
        environment: browser.environment as never,
        onValidated: (value) => validated.push(value),
      },
    )).rejects.toThrow("review dependencies");

    expect(validated).toEqual([validatedReturn]);
    expect(api.cancelWorkflowEditor).not.toHaveBeenCalled();
  });

  it("retains an exact submission and live canvas when consume is ambiguous", async () => {
    const api = fakeApi({
      consumeWorkflowEditor: vi.fn().mockRejectedValue(new Error("response was lost")),
    });
    const browser = harness({ save: { graph: { nodes: [1] }, prompt: { 1: {} } } });
    const submissions: Array<unknown> = [];

    await expect(runWorkflowEditor(
      api,
      "workflow-1",
      browser.popup,
      {
        environment: browser.environment as never,
        onSubmission: (value) => submissions.push(value),
      },
    )).rejects.toThrow("response was lost");

    expect(submissions).toEqual([{
      workflowId: "workflow-1",
      sessionId: "session-1",
      nonce: "nonce-1",
      baseRevisionId: "revision-1",
      uiGraph: { nodes: [1] },
      apiPrompt: { 1: {} },
    }]);
    expect(api.cancelWorkflowEditor).not.toHaveBeenCalled();
    expect(browser.popup.close).not.toHaveBeenCalled();
  });

  it("keeps the live canvas after an invalid save response", async () => {
    const api = fakeApi();
    const browser = harness({
      save: { graph: { nodes: [1] }, prompt: { 1: {} } },
      saveNonce: "another-session",
    });

    await expect(runWorkflowEditor(
      api,
      "workflow-1",
      browser.popup,
      { environment: browser.environment as never },
    )).rejects.toThrow("window remains open");

    expect(browser.popup.close).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(api.cancelWorkflowEditor).toHaveBeenCalled());
  });

  it("rejects a shell ready message from the wrong origin", async () => {
    const api = fakeApi();
    const browser = harness({ readyOrigin: "http://127.0.0.1:9999" });
    const run = runWorkflowEditor(
      api,
      "workflow-1",
      browser.popup,
      { environment: browser.environment as never },
    );
    await Promise.resolve();
    browser.fireTimeout();

    await expect(run).rejects.toThrow("did not become ready");
    await vi.waitFor(() => expect(api.cancelWorkflowEditor).toHaveBeenCalled());
  });

  it("cancels when the popup closes before a save", async () => {
    const api = fakeApi();
    const browser = harness();
    const run = runWorkflowEditor(
      api,
      "workflow-1",
      browser.popup,
      { environment: browser.environment as never },
    );
    await vi.waitFor(() => expect(browser.popup.postMessage).toHaveBeenCalled());
    browser.popup.closed = true;
    browser.firePoll();

    await expect(run).rejects.toThrow("workflow editor was closed");
    await vi.waitFor(() => expect(api.cancelWorkflowEditor).toHaveBeenCalled());
  });

  it("warns before a long editing session expires", async () => {
    const api = fakeApi({
      startWorkflowEditor: vi.fn().mockResolvedValue(session({
        expires_at: new Date(Date.now() + 10 * 60_000).toISOString(),
      })),
    });
    const browser = harness();
    const controller = new AbortController();
    const warnings: string[] = [];
    const phases: string[] = [];
    const run = runWorkflowEditor(api, "workflow-1", browser.popup, {
      environment: browser.environment as never,
      signal: controller.signal,
      onPhase: (phase) => phases.push(phase),
      onExpiryWarning: (expiresAt) => warnings.push(expiresAt),
    });
    await vi.waitFor(() => expect(phases).toContain("editing"));
    browser.fireTimeout();
    expect(warnings).toHaveLength(1);
    controller.abort();
    await expect(run).rejects.toThrow("cancelled");
  });

  it("keeps the live ComfyUI canvas open when editor authority expires", async () => {
    const api = fakeApi();
    const browser = harness();
    const phases: string[] = [];
    const run = runWorkflowEditor(api, "workflow-1", browser.popup, {
      environment: browser.environment as never,
      onPhase: (phase) => phases.push(phase),
    });
    await vi.waitFor(() => expect(phases).toContain("editing"));
    browser.fireTimeout();
    await expect(run).rejects.toThrow("secure editor session expired");
    expect(browser.popup.close).not.toHaveBeenCalled();
    expect(api.cancelWorkflowEditor).toHaveBeenCalledWith(
      "workflow-1",
      "session-1",
      "nonce-1",
    );
  });
});
