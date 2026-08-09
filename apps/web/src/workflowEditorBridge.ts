import type {
  WorkflowEditorDraft,
  WorkflowEditorReturn,
  WorkflowEditorSession,
} from "./types";

const PROTOCOL_VERSION = 2;
const BRIDGE_SOURCE = "lm-atelier-workflow-editor";
const PARENT_SOURCE = "lm-atelier";
const SHELL_URL = "/api/workflow-editor/shell";
const CONNECT_TIMEOUT_MS = 30_000;
const LOAD_TIMEOUT_MS = 30_000;
const CLOSE_POLL_MS = 250;
const EXPIRY_WARNING_MS = 5 * 60_000;

type JsonRecord = Record<string, unknown>;

export type WorkflowEditorPhase =
  | "preparing"
  | "connecting"
  | "loading"
  | "editing"
  | "validating"
  | "saving";

export type WorkflowEditorOutcome =
  | { kind: "unchanged"; returned: WorkflowEditorReturn }
  | { kind: "draft"; returned: WorkflowEditorReturn; draft: WorkflowEditorDraft };

export interface WorkflowEditorSubmission {
  workflowId: string;
  sessionId: string;
  nonce: string;
  baseRevisionId: string;
  uiGraph: JsonRecord;
  apiPrompt: JsonRecord;
}

export interface WorkflowEditorApi {
  startWorkflowEditor(id: string): Promise<WorkflowEditorSession>;
  consumeWorkflowEditor(
    workflowId: string,
    sessionId: string,
    payload: {
      nonce: string;
      base_revision_id: string;
      ui_graph: JsonRecord;
      api_prompt: JsonRecord;
    },
  ): Promise<WorkflowEditorReturn>;
  createWorkflowEditorDraft(
    workflowId: string,
    validatedReturnId: string,
  ): Promise<WorkflowEditorDraft>;
  cancelWorkflowEditor(workflowId: string, sessionId: string, nonce: string): Promise<void>;
}

interface PopupWindow {
  readonly closed: boolean;
  close(): void;
  postMessage(message: unknown, targetOrigin: string, transfer?: Transferable[]): void;
}

interface EditorEnvironment {
  readonly origin: string;
  addMessageListener(listener: (event: MessageEvent) => void): void;
  removeMessageListener(listener: (event: MessageEvent) => void): void;
  createChannel(): MessageChannel;
  setTimer(callback: () => void, delay: number): number;
  clearTimer(timer: number): void;
  setPoll(callback: () => void, delay: number): number;
  clearPoll(timer: number): void;
}

interface RunOptions {
  signal?: AbortSignal;
  onPhase?: (phase: WorkflowEditorPhase) => void;
  onValidated?: (result: WorkflowEditorReturn | null) => void;
  onSubmission?: (submission: WorkflowEditorSubmission | null) => void;
  onExpiryWarning?: (expiresAt: string) => void;
  environment?: EditorEnvironment;
}

class WorkflowEditorSessionExpiredError extends Error {}

function browserEnvironment(): EditorEnvironment {
  return {
    origin: window.location.origin,
    addMessageListener: (listener) => window.addEventListener("message", listener),
    removeMessageListener: (listener) => window.removeEventListener("message", listener),
    createChannel: () => new MessageChannel(),
    setTimer: (callback, delay) => window.setTimeout(callback, delay),
    clearTimer: (timer) => window.clearTimeout(timer),
    setPoll: (callback, delay) => window.setInterval(callback, delay),
    clearPoll: (timer) => window.clearInterval(timer),
  };
}

function isRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isEnvelope(value: unknown, type: string): value is JsonRecord {
  return isRecord(value)
    && value.source === BRIDGE_SOURCE
    && value.protocol === PROTOCOL_VERSION
    && value.type === type;
}

function isBridgeEnvelope(value: unknown): value is JsonRecord {
  return isRecord(value)
    && value.source === BRIDGE_SOURCE
    && value.protocol === PROTOCOL_VERSION
    && typeof value.type === "string"
    && ["connected", "loaded", "save", "error"].includes(value.type);
}

function abortError(message: string): Error {
  return new DOMException(message, "AbortError");
}

function preservedCanvasError(error: unknown): Error {
  const message = isRecord(error) && typeof error.message === "string"
    ? error.message
    : "Workflow editing failed.";
  return new Error(
    `${message} The ComfyUI window remains open so you can correct or download the workflow.`,
    { cause: error },
  );
}

function remainingSessionTime(session: WorkflowEditorSession): number {
  const expiresAt = Date.parse(session.expires_at);
  return Number.isFinite(expiresAt) ? Math.max(1, expiresAt - Date.now()) : 1;
}

function waitForShell(
  popup: PopupWindow,
  environment: EditorEnvironment,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      environment.removeMessageListener(onMessage);
      environment.clearTimer(timeout);
      environment.clearPoll(closePoll);
      signal?.removeEventListener("abort", onAbort);
      if (error) reject(error);
      else resolve();
    };
    const onMessage = (event: MessageEvent) => {
      if (
        event.source === popup
        && event.origin === environment.origin
        && event.ports.length === 0
        && isEnvelope(event.data, "ready")
      ) {
        finish();
      }
    };
    const onAbort = () => finish(abortError("Workflow editing was cancelled."));
    const timeout = environment.setTimer(
      () => finish(new Error("ComfyUI did not become ready for workflow editing.")),
      CONNECT_TIMEOUT_MS,
    );
    const closePoll = environment.setPoll(() => {
      if (popup.closed) finish(abortError("The workflow editor was closed."));
    }, CLOSE_POLL_MS);
    environment.addMessageListener(onMessage);
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
  });
}

class PortInbox {
  private readonly queue: Array<JsonRecord | Error> = [];
  private notify: (() => void) | null = null;

  constructor(
    private readonly port: MessagePort,
    private readonly popup: PopupWindow,
    private readonly environment: EditorEnvironment,
    private readonly signal?: AbortSignal,
  ) {
    port.onmessage = (event) => {
      if (!isBridgeEnvelope(event.data)) {
        this.queue.push(new Error("ComfyUI returned an invalid workflow editor message."));
      } else if (event.data.type === "error") {
        const code = typeof event.data.code === "string" ? event.data.code : "unknown-error";
        this.queue.push(new Error(`ComfyUI workflow editor failed: ${code}`));
      } else {
        this.queue.push(event.data);
      }
      this.notify?.();
    };
    port.onmessageerror = () => {
      this.queue.push(new Error("ComfyUI could not return the workflow editor message."));
      this.notify?.();
    };
    port.start();
  }

  wait(expectedType: string, timeoutMs: number, timeoutError?: Error): Promise<JsonRecord> {
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = (value?: JsonRecord, error?: Error) => {
        if (settled) return;
        settled = true;
        this.notify = null;
        this.environment.clearTimer(timeout);
        this.environment.clearPoll(closePoll);
        this.signal?.removeEventListener("abort", onAbort);
        if (error) reject(error);
        else resolve(value!);
      };
      const inspect = () => {
        const next = this.queue.shift();
        if (!next) return;
        if (next instanceof Error) {
          finish(undefined, next);
          return;
        }
        if (next.type !== expectedType) {
          finish(undefined, new Error(`ComfyUI sent ${String(next.type)} while ${expectedType} was expected.`));
          return;
        }
        finish(next);
      };
      const onAbort = () => finish(undefined, abortError("Workflow editing was cancelled."));
      const timeout = this.environment.setTimer(
        () => finish(
          undefined,
          timeoutError ?? new Error(`Timed out while waiting for workflow editor ${expectedType}.`),
        ),
        timeoutMs,
      );
      const closePoll = this.environment.setPoll(() => {
        if (this.popup.closed) finish(undefined, abortError("The workflow editor was closed."));
      }, CLOSE_POLL_MS);
      this.notify = inspect;
      this.signal?.addEventListener("abort", onAbort, { once: true });
      if (this.signal?.aborted) onAbort();
      else inspect();
    });
  }

  close(): void {
    this.notify = null;
    this.port.onmessage = null;
    this.port.onmessageerror = null;
    this.port.close();
  }
}

function postPort(port: MessagePort, type: string, details: JsonRecord = {}): void {
  port.postMessage({
    source: PARENT_SOURCE,
    protocol: PROTOCOL_VERSION,
    type,
    ...details,
  });
}

async function bestEffortCancel(
  api: WorkflowEditorApi,
  workflowId: string,
  session: WorkflowEditorSession | null,
): Promise<void> {
  if (!session) return;
  try {
    await api.cancelWorkflowEditor(workflowId, session.id, session.nonce);
  } catch {
    // The original failure is more useful than a cleanup failure. The server
    // also expires short-lived sessions and invalidates them on runtime exit.
  }
}

export function openWorkflowEditorPopup(): Window | null {
  return window.open(SHELL_URL, "_blank", "popup");
}

export async function runWorkflowEditor(
  api: WorkflowEditorApi,
  workflowId: string,
  popup: PopupWindow,
  options: RunOptions = {},
): Promise<WorkflowEditorOutcome> {
  const environment = options.environment ?? browserEnvironment();
  let session: WorkflowEditorSession | null = null;
  let consumed = false;
  let submitted = false;
  let editingStarted = false;
  let keepPopupOpen = false;
  let inbox: PortInbox | null = null;
  let expiryWarningTimer: number | null = null;
  options.onPhase?.("preparing");
  const start = api.startWorkflowEditor(workflowId);
  try {
    const [started] = await Promise.all([
      start,
      waitForShell(popup, environment, options.signal),
    ]);
    session = started;
    if (session.protocol_version !== PROTOCOL_VERSION) {
      throw new Error("This ComfyUI editor bridge version is not supported.");
    }
    options.onPhase?.("connecting");
    const channel = environment.createChannel();
    inbox = new PortInbox(channel.port1, popup, environment, options.signal);
    const connected = inbox.wait("connected", CONNECT_TIMEOUT_MS);
    popup.postMessage(
      { source: PARENT_SOURCE, protocol: PROTOCOL_VERSION, type: "connect" },
      environment.origin,
      [channel.port2],
    );
    await connected;

    options.onPhase?.("loading");
    const loaded = inbox.wait("loaded", LOAD_TIMEOUT_MS);
    postPort(channel.port1, "load", { nonce: session.nonce, graph: session.ui_graph });
    const loadedMessage = await loaded;
    if (loadedMessage.nonce !== session.nonce) {
      throw new Error("ComfyUI acknowledged a different workflow editor session.");
    }

    editingStarted = true;
    options.onPhase?.("editing");
    const remaining = remainingSessionTime(session);
    if (remaining > EXPIRY_WARNING_MS) {
      expiryWarningTimer = environment.setTimer(
        () => options.onExpiryWarning?.(session!.expires_at),
        remaining - EXPIRY_WARNING_MS,
      );
    }
    const saved = await inbox.wait(
      "save",
      remaining,
      new WorkflowEditorSessionExpiredError(
        "The secure editor session expired. The ComfyUI window remains open so you can download the workflow before starting again.",
      ),
    );
    if (expiryWarningTimer !== null) {
      environment.clearTimer(expiryWarningTimer);
      expiryWarningTimer = null;
    }
    if (
      saved.nonce !== session.nonce
      || !isRecord(saved.graph)
      || !isRecord(saved.prompt)
    ) {
      throw new Error("ComfyUI returned an invalid workflow editor save.");
    }

    options.onPhase?.("validating");
    const submission: WorkflowEditorSubmission = {
      workflowId,
      sessionId: session.id,
      nonce: session.nonce,
      baseRevisionId: session.base_revision_id,
      uiGraph: saved.graph,
      apiPrompt: saved.prompt,
    };
    submitted = true;
    options.onSubmission?.(submission);
    const returned = await api.consumeWorkflowEditor(workflowId, session.id, {
      nonce: submission.nonce,
      base_revision_id: submission.baseRevisionId,
      ui_graph: submission.uiGraph,
      api_prompt: submission.apiPrompt,
    });
    consumed = true;
    options.onSubmission?.(null);
    options.onValidated?.(returned);
    if (!returned.changed) return { kind: "unchanged", returned };

    options.onPhase?.("saving");
    const draft = await api.createWorkflowEditorDraft(
      workflowId,
      returned.validated_return_id,
    );
    options.onValidated?.(null);
    return { kind: "draft", returned, draft };
  } catch (error) {
    keepPopupOpen = !popup.closed && (
      editingStarted
      || error instanceof WorkflowEditorSessionExpiredError
      || (submitted && !consumed)
    );
    if (!consumed && !submitted) {
      void start.then((started) => bestEffortCancel(api, workflowId, session ?? started)).catch(() => {});
    }
    if (keepPopupOpen && !(error instanceof WorkflowEditorSessionExpiredError)) {
      throw preservedCanvasError(error);
    }
    throw error;
  } finally {
    if (expiryWarningTimer !== null) environment.clearTimer(expiryWarningTimer);
    inbox?.close();
    if (!keepPopupOpen && !popup.closed) popup.close();
  }
}
