import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

function responses(value: unknown) {
  const fetch = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(value), { status: 200 }));
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

it("reads only metadata and forwards cancellation to that request", async () => {
  const fetch = responses({ id: "chat/one", title: "Harbor sketches" });
  const { api } = await import("./api");
  const controller = new AbortController();

  await expect(api.chatMetadata("chat/one", controller.signal)).resolves.toMatchObject({ id: "chat/one" });

  expect(fetch.mock.calls[1][0]).toBe("/api/chats/chat%2Fone/metadata");
  expect(fetch.mock.calls[1][1]?.signal).toBe(controller.signal);
});

it("requests a bounded branch page with its anchor and cancellation signal", async () => {
  const page = { chat_id: "chat/one", messages: [], has_older: false, has_newer: true };
  const fetch = responses(page);
  const { api } = await import("./api");
  const controller = new AbortController();

  await expect(api.chatMessages("chat/one", {
    headId: "head&one", before: "older/one", signal: controller.signal,
  })).resolves.toEqual(page);

  expect(fetch.mock.calls[1][0]).toBe("/api/chats/chat%2Fone/messages?limit=40&head_id=head%26one&before=older%2Fone");
  expect(fetch.mock.calls[1][1]?.signal).toBe(controller.signal);
});

it("reads branch-wide context without requesting the full transcript", async () => {
  const context = { chat_id: "chat/one", head_id: "head&one", has_prior_image: true,
    has_prior_visual: true, has_pending_response: false };
  const fetch = responses(context);
  const { api } = await import("./api");
  const controller = new AbortController();

  await expect(api.chatContext("chat/one", "head&one", controller.signal)).resolves.toEqual(context);

  expect(fetch.mock.calls[1][0]).toBe("/api/chats/chat%2Fone/context?head_id=head%26one");
  expect(fetch.mock.calls[1][1]?.signal).toBe(controller.signal);
});

it("bounds search history to the displayed branch and message range", async () => {
  const page = { chat_id: "chat/one", searches: [], next_before: "run/older" };
  const fetch = responses(page);
  const { api } = await import("./api");
  const controller = new AbortController();

  await expect(api.chatSearches("chat/one", {
    headId: "head&one", oldestMessageId: "message/old", before: "run&last", signal: controller.signal,
  })).resolves.toEqual(page);

  const url = new URL(fetch.mock.calls[1][0], "http://localhost");
  expect(url.pathname).toBe("/api/chats/chat%2Fone/searches");
  expect(Object.fromEntries(url.searchParams)).toEqual({
    limit: "40", head_id: "head&one", oldest_message_id: "message/old", before: "run&last",
  });
  expect(fetch.mock.calls[1][1]?.signal).toBe(controller.signal);
});

it("pages pending search decisions across all branches", async () => {
  const page = { chat_id: "chat/one", searches: [], next_before: null };
  const fetch = responses(page);
  const { api } = await import("./api");

  await expect(api.chatSearches("chat/one", {
    pendingOnly: true, limit: 12, before: "run/last",
  })).resolves.toEqual(page);

  const url = new URL(fetch.mock.calls[1][0], "http://localhost");
  expect(Object.fromEntries(url.searchParams)).toEqual({ limit: "12", pending_only: "true", before: "run/last" });
});

it("reads a bounded page of edit sources with an encoded result and cancellation", async () => {
  const page = { chat_id: "chat/one", result_message_id: "result/one", steps: [], next_before: null };
  const fetch = responses(page);
  const { api } = await import("./api");
  const controller = new AbortController();

  await expect(api.chatEditLineage("chat/one", "result/one", {
    before: "user&older", limit: 2, signal: controller.signal,
  })).resolves.toEqual(page);

  const url = new URL(fetch.mock.calls[1][0], "http://localhost");
  expect(url.pathname).toBe("/api/chats/chat%2Fone/messages/result%2Fone/lineage");
  expect(Object.fromEntries(url.searchParams)).toEqual({ limit: "2", before: "user&older" });
  expect(fetch.mock.calls[1][1]?.signal).toBe(controller.signal);
});
