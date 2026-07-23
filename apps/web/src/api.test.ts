import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
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
  await api.branchMessage("message-user", "Count to 1000", { max_tokens: 4096 });
  await api.regenerateMessage("message-assistant", { max_tokens: 4096 });

  expect(fetchMock.mock.calls[1][0]).toBe("/api/messages/message-user/branch");
  expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
    text: "Count to 1000",
    settings: { max_tokens: 4096 },
  });
  expect(fetchMock.mock.calls[2][0]).toBe("/api/messages/message-assistant/regenerate");
  expect(JSON.parse(String(fetchMock.mock.calls[2][1]?.body))).toEqual({
    settings: { max_tokens: 4096 },
  });
});
