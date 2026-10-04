import { afterEach, expect, it, vi } from "vitest";
import { workspaceLockApi } from "./workspaceLockApi";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.resetModules();
});

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function session(lock: { workspace_locked: boolean; lock_epoch: string | null }): Response {
  return json({ csrf_token: "csrf", event_epoch: "events", event_sequence: 0, ...lock });
}

/** The real client and the lock state it reports into, loaded fresh for each test. */
async function load() {
  const client = await import("./api");
  const lock = await import("./workspaceLockState");
  return { ...client, lock };
}

function sentHeaders(fetchMock: ReturnType<typeof vi.fn>, call: number): Headers {
  const init = fetchMock.mock.calls[call][1] as RequestInit;
  return new Headers(init.headers);
}

class FakeWebSocket {
  static opened = 0;
  onopen: (() => void) | null = null;
  onmessage: ((message: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  constructor(readonly url = "") {
    FakeWebSocket.opened += 1;
    sockets.push(this);
  }
  close() {}
}
const sockets: FakeWebSocket[] = [];

it("sends only a PIN in the body of an unlock, and never in its address", async () => {
  const request = vi.fn().mockResolvedValue({});
  const lockApi = workspaceLockApi(request);
  await lockApi.unlockWorkspace("4321");
  expect(request).toHaveBeenLastCalledWith("/api/privacy/unlock", { method: "POST", body: JSON.stringify({ pin: "4321" }) });
  await lockApi.unlockWorkspace();
  expect(request).toHaveBeenLastCalledWith("/api/privacy/unlock", { method: "POST", body: "{}" });
  const write = { expected_revision: 3, enabled: true };
  await lockApi.updateWorkspaceLockPolicy(write);
  expect(request).toHaveBeenLastCalledWith("/api/privacy/policy", { method: "PUT", body: JSON.stringify(write) });
});

it("locks this window when the server refuses a request because the workspace is locked", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(session({ workspace_locked: false, lock_epoch: null }))
    .mockResolvedValueOnce(json({ detail: "LM Atelier is locked.", code: "workspace-locked" }, 423));
  vi.stubGlobal("fetch", fetchMock);
  const { api, ApiError, lock } = await load();

  const refusal = await api.chats().catch((error: unknown) => error);

  expect(refusal).toBeInstanceOf(ApiError);
  expect((refusal as InstanceType<typeof ApiError>).status).toBe(423);
  expect(lock.isWorkspaceLockBlocking()).toBe(true);
  expect(lock.isWorkspaceLockRefusal(refusal)).toBe(true);
  // Refused once, and not asked again: a locked workspace answers a retry the same way.
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it("starts over when the server says the lock changed since this window last looked", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(session({ workspace_locked: false, lock_epoch: "epoch-one" }))
    .mockResolvedValueOnce(json({ detail: "The workspace lock changed. Reload to continue.", code: "workspace-lock-changed" }, 423));
  vi.stubGlobal("fetch", fetchMock);
  const { api, lock } = await load();

  await expect(api.chats()).rejects.toMatchObject({ status: 423 });

  expect(lock.workspaceLockView().phase).toBe("changed");
  // The epoch it held is kept until a fresh read says otherwise.
  expect(lock.workspaceLockEpoch()).toBe("epoch-one");
});

it("sends the lock epoch only once one is known", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(session({ workspace_locked: false, lock_epoch: null }))
    .mockResolvedValueOnce(json([]))
    .mockResolvedValueOnce(json([]));
  vi.stubGlobal("fetch", fetchMock);
  const { api, lock } = await load();

  await api.chats();
  expect(sentHeaders(fetchMock, 1).has("x-local-lm-lock-epoch")).toBe(false);

  lock.applyWorkspaceLockStatus({ locked: false, enabled: true, require_pin: false, lock_epoch: "epoch-one" });
  await api.chats();
  expect(sentHeaders(fetchMock, 2).get("x-local-lm-lock-epoch")).toBe("epoch-one");
});

it("sends the epoch a session reported, but not to the session request itself", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(session({ workspace_locked: false, lock_epoch: "epoch-one" }))
    .mockResolvedValueOnce(json([]));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await load();

  await api.chats();

  expect(fetchMock.mock.calls[0][0]).toBe("/api/session");
  expect(sentHeaders(fetchMock, 0).has("x-local-lm-lock-epoch")).toBe(false);
  expect(sentHeaders(fetchMock, 1).get("x-local-lm-lock-epoch")).toBe("epoch-one");
});

it("keeps the epoch it had when a renewed session reports a different one", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(session({ workspace_locked: false, lock_epoch: "epoch-one" }))
    .mockResolvedValueOnce(json({ detail: "Session required", code: "session-required" }, 401))
    .mockResolvedValueOnce(session({ workspace_locked: false, lock_epoch: "epoch-two" }))
    .mockResolvedValueOnce(json([]));
  vi.stubGlobal("fetch", fetchMock);
  const { api, lock } = await load();

  await api.chats();

  expect(lock.workspaceLockEpoch()).toBe("epoch-one");
  expect(lock.isWorkspaceLockBlocking()).toBe(true);
  expect(sentHeaders(fetchMock, 3).get("x-local-lm-lock-epoch")).toBe("epoch-one");
});

it.each([
  [403, { detail: "The workspace could not be unlocked.", code: "workspace-pin-refused", retry_after_seconds: 0 }],
  [429, { detail: "Too many attempts.", code: "workspace-pin-throttled", retry_after_seconds: 4 }],
])("sends a refused unlock exactly once (%i)", async (status, body) => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(session({ workspace_locked: true, lock_epoch: "epoch-one" }))
    .mockResolvedValueOnce(json(body, status));
  vi.stubGlobal("fetch", fetchMock);
  const { api, lock } = await load();

  const refusal = await api.unlockWorkspace("4321").catch((error: unknown) => error);

  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/privacy/unlock");
  expect(lock.workspaceLockFailure(refusal)).toEqual({
    status, code: body.code, retryAfterSeconds: body.retry_after_seconds,
  });
});

it("opens no live connection while a new session says the workspace is locked", async () => {
  vi.useFakeTimers();
  sockets.length = 0;
  FakeWebSocket.opened = 0;
  const fetchMock = vi.fn().mockImplementation(async () => session({ workspace_locked: true, lock_epoch: "epoch-one" }));
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);
  const { connectEvents, lock } = await load();
  const onStatus = vi.fn();

  const dispose = await connectEvents(vi.fn(), onStatus);
  await vi.advanceTimersByTimeAsync(10_000);

  expect(FakeWebSocket.opened).toBe(0);
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(lock.isWorkspaceLockBlocking()).toBe(true);
  dispose();
});

it("stops reconnecting when the server closes the connection because the workspace locked", async () => {
  vi.useFakeTimers();
  sockets.length = 0;
  FakeWebSocket.opened = 0;
  const fetchMock = vi.fn().mockImplementation(async () => session({ workspace_locked: false, lock_epoch: "epoch-one" }));
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);
  const { connectEvents, lock } = await load();
  const onStatus = vi.fn();

  const dispose = await connectEvents(vi.fn(), onStatus);
  expect(sockets).toHaveLength(1);
  sockets[0].onclose?.({ code: 4423 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(10_000);

  expect(lock.isWorkspaceLockBlocking()).toBe(true);
  expect(onStatus).toHaveBeenLastCalledWith(false);
  expect(sockets).toHaveLength(1);
  expect(fetchMock).toHaveBeenCalledTimes(1);
  dispose();
});

it("still reconnects after any other close", async () => {
  vi.useFakeTimers();
  sockets.length = 0;
  FakeWebSocket.opened = 0;
  const fetchMock = vi.fn().mockImplementation(async () => session({ workspace_locked: false, lock_epoch: "epoch-one" }));
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);
  const { connectEvents, lock } = await load();

  const dispose = await connectEvents(vi.fn(), vi.fn());
  sockets[0].onclose?.({ code: 1006 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);

  expect(lock.isWorkspaceLockBlocking()).toBe(false);
  expect(sockets).toHaveLength(2);
  dispose();
});

it("names the lock epoch it holds when it opens a live connection, and only then", async () => {
  sockets.length = 0;
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => session({ workspace_locked: false, lock_epoch: "epoch one" })));
  const named = await load();
  const disposeNamed = await named.connectEvents(vi.fn(), vi.fn());
  disposeNamed();
  vi.resetModules();
  vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => session({ workspace_locked: false, lock_epoch: null })));
  const unnamed = await load();
  const disposeUnnamed = await unnamed.connectEvents(vi.fn(), vi.fn());
  disposeUnnamed();

  expect(sockets.map((socket) => socket.url.replace(/^.*\/api\/events/, "/api/events"))).toEqual([
    "/api/events?after=0&lock_epoch=epoch%20one",
    "/api/events?after=0",
  ]);
});
