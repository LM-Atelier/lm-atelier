import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.resetModules();
  sessionStorage.clear();
  localStorage.clear();
});

it("keeps the opaque private token in session storage and scopes content requests", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({
        token: "scope_private_token",
        event_epoch: "private-epoch",
        event_sequence: 0,
        disclosure: "Private session disclosure",
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify([]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(new Blob(["private image bytes"], { type: "image/png" }), {
        status: 200,
        headers: { "content-type": "image/png" },
      }),
    )
    .mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.startIncognito();
  expect(sessionStorage.getItem("lm-atelier-incognito-session")).toBe(
    "scope_private_token",
  );
  expect(localStorage.length).toBe(0);
  await api.chats();
  await api.artifactBlob("sha256:private");
  await api.endIncognito();

  const startHeaders = new Headers(fetchMock.mock.calls[1][1]?.headers);
  const chatHeaders = new Headers(fetchMock.mock.calls[2][1]?.headers);
  const artifactHeaders = new Headers(fetchMock.mock.calls[3][1]?.headers);
  const endHeaders = new Headers(fetchMock.mock.calls[4][1]?.headers);
  expect(startHeaders.has("x-lm-atelier-incognito")).toBe(false);
  expect(chatHeaders.get("x-lm-atelier-incognito")).toBe("scope_private_token");
  expect(artifactHeaders.get("x-lm-atelier-incognito")).toBe("scope_private_token");
  expect(endHeaders.get("x-lm-atelier-incognito")).toBe("scope_private_token");
  expect(fetchMock.mock.calls[3][1]?.cache).toBe("no-store");
  expect(sessionStorage.getItem("lm-atelier-incognito-session")).toBeNull();
  expect(localStorage.length).toBe(0);
});

it("opens the event socket from the sequence returned by session initialization", async () => {
  const urls: string[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: (() => void) | null = null;

    constructor(url: string) {
      urls.push(url);
    }

    close() {}
  }

  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ csrf_token: "csrf", event_sequence: 742 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    ),
  );
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());

  expect(urls).toHaveLength(1);
  expect(new URL(urls[0]).searchParams.get("after")).toBe("742");
  dispose();
});

it("sends turn overrides with edited branches and regenerated responses", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({}), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    ));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.branchMessage(
    "message-user",
    "Count to 1000",
    "text",
    { max_tokens: 4096 },
  );
  await api.regenerateMessage("message-assistant", { max_tokens: 4096 });

  expect(fetchMock.mock.calls[1][0]).toBe("/api/messages/message-user/branch");
  expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
    text: "Count to 1000",
    mode: "text",
    input_artifact_ids: [],
    settings: { max_tokens: 4096 },
  });
  expect(fetchMock.mock.calls[2][0]).toBe("/api/messages/message-assistant/regenerate");
  expect(JSON.parse(String(fetchMock.mock.calls[2][1]?.body))).toEqual({
    settings: { max_tokens: 4096 },
  });
});

it("uses the explicit stop-and-send endpoint and preserves its idempotency key", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({}), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.stopAndSendTurn(
    "chat/one",
    "Use this instead",
    "text",
    [],
    { max_tokens: 128 },
    "client-turn-7",
  );

  expect(fetchMock.mock.calls[1][0]).toBe("/api/chats/chat/one/stop-and-send");
  expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
    text: "Use this instead",
    mode: "text",
    idempotency_key: "client-turn-7",
  });
});

it("uses the recovery and unsuccessful-job action contracts", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ id: "job/retry" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ name: "backup one.sqlite3" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ name: "backup one.sqlite3", restore_pending: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.retryJob("job/retry");
  await api.verifyBackup("backup one.sqlite3");
  await api.restoreBackup("backup one.sqlite3");
  await api.deleteBackup("backup one.sqlite3");

  expect(fetchMock.mock.calls.slice(1).map(([url, init]) => [url, init?.method])).toEqual([
    ["/api/jobs/job%2Fretry/retry", "POST"],
    ["/api/backups/backup%20one.sqlite3/verify", "POST"],
    ["/api/backups/backup%20one.sqlite3/restore", "POST"],
    ["/api/backups/backup%20one.sqlite3", "DELETE"],
  ]);
});

it("requests transactional profile cleanup when deleting an installed model", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.deleteModel("model-1", true);

  expect(fetchMock.mock.calls[1][0]).toBe(
    "/api/models/model-1?delete_profiles=true",
  );
  expect(fetchMock.mock.calls[1][1]?.method).toBe("DELETE");
});

it("retries session initialization after a transient startup failure", async () => {
  const fetchMock = vi.fn()
    .mockRejectedValueOnce(new Error("service is starting"))
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-recovered" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify([]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.projects()).rejects.toThrow("service is starting");
  await expect(api.projects()).resolves.toEqual([]);

  expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
    "/api/session",
    "/api/session",
    "/api/projects?include_archived=false&query=",
  ]);
});

it("refreshes an expired session once before retrying an API request", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-old" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "session required" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-new" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ id: "project-1", name: "Recovered" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.createProject("Recovered")).resolves.toMatchObject({
    id: "project-1",
  });

  const firstHeaders = new Headers(fetchMock.mock.calls[1][1]?.headers);
  const retriedHeaders = new Headers(fetchMock.mock.calls[3][1]?.headers);
  expect(firstHeaders.get("x-local-lm-csrf")).toBe("csrf-old");
  expect(retriedHeaders.get("x-local-lm-csrf")).toBe("csrf-new");
});

it("keeps retrying event initialization while the local service starts", async () => {
  vi.useFakeTimers();
  const urls: string[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor(url: string) {
      urls.push(url);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockRejectedValueOnce(new Error("service is starting"))
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf", event_sequence: 9 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const onStatus = vi.fn();
  const dispose = await connectEvents(vi.fn(), onStatus);

  expect(urls).toHaveLength(0);
  expect(onStatus).toHaveBeenCalledWith(false);
  await vi.advanceTimersByTimeAsync(1_000);
  expect(urls).toHaveLength(1);
  expect(new URL(urls[0]).searchParams.get("after")).toBe("9");
  dispose();
});

it("renews the session after an authenticated event socket is rejected", async () => {
  vi.useFakeTimers();
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor() {
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-old" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-new" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());
  expect(sockets).toHaveLength(1);

  sockets[0].onclose?.({ code: 4401 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);

  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(sockets).toHaveLength(2);
  dispose();
});

it("replays events from zero when the service sequence resets after a restart", async () => {
  vi.useFakeTimers();
  const urls: string[] = [];
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor(url: string) {
      urls.push(url);
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-old", event_sequence: 742 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-new", event_sequence: 3 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());
  expect(new URL(urls[0]).searchParams.get("after")).toBe("742");

  sockets[0].onclose?.({ code: 1006 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);

  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(new URL(urls[1]).searchParams.get("after")).toBe("0");
  dispose();
});

it("replays from zero when a restarted service has already advanced beyond the old sequence", async () => {
  vi.useFakeTimers();
  const urls: string[] = [];
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor(url: string) {
      urls.push(url);
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({
        csrf_token: "csrf-old",
        event_epoch: "old-process",
        event_sequence: 3,
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({
        csrf_token: "csrf-new",
        event_epoch: "new-process",
        event_sequence: 15,
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());
  expect(new URL(urls[0]).searchParams.get("after")).toBe("3");

  sockets[0].onclose?.({ code: 1006 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);

  expect(new URL(urls[1]).searchParams.get("after")).toBe("0");
  dispose();
});

it("retains the last received sequence during a same-service reconnect", async () => {
  vi.useFakeTimers();
  const urls: string[] = [];
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor(url: string) {
      urls.push(url);
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf", event_sequence: 10 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf", event_sequence: 15 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());
  sockets[0].onmessage?.({
    data: JSON.stringify({ sequence: 12, type: "generation.progress", payload: {} }),
  } as MessageEvent);
  sockets[0].onclose?.({ code: 1006 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);

  expect(new URL(urls[1]).searchParams.get("after")).toBe("12");
  dispose();
});

it("notifies the client after each event socket reconnect, not the initial open", async () => {
  vi.useFakeTimers();
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor() {
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-old", event_sequence: 10 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-new", event_sequence: 10 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const onReconnect = vi.fn();
  const dispose = await connectEvents(vi.fn(), vi.fn(), onReconnect);
  sockets[0].onopen?.();
  expect(onReconnect).not.toHaveBeenCalled();

  sockets[0].onclose?.({ code: 1006 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);
  sockets[1].onopen?.();
  expect(onReconnect).toHaveBeenCalledTimes(1);
  dispose();
});
