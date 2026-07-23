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
