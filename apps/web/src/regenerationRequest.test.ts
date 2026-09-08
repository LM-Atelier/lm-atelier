import { beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { regenerateWithRetry } from "./regenerationRequest";
import type { TurnAccepted } from "./types";

vi.mock("./api", () => ({ api: { regenerateMessage: vi.fn() } }));

const accepted = { run: { id: "accepted-run" } } as TurnAccepted;
beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
  vi.mocked(api.regenerateMessage).mockReset();
});

it("reuses an unacknowledged request after the client module reloads", async () => {
  vi.mocked(api.regenerateMessage).mockRejectedValueOnce(new Error("Connection lost"));
  await expect(regenerateWithRetry("chat-one", "message-one", { temperature: 0.25 }))
    .rejects.toThrow("Connection lost");
  const firstKey = vi.mocked(api.regenerateMessage).mock.calls[0][2];
  expect(firstKey).toEqual(expect.any(String));
  expect(localStorage.length).toBe(1);
  expect(localStorage.key(0)).not.toContain("message-one");
  vi.resetModules();
  const reloaded = await import("./regenerationRequest");
  vi.mocked(api.regenerateMessage).mockResolvedValueOnce(accepted);
  await expect(reloaded.regenerateWithRetry("chat-one", "message-one", { temperature: 0.25 }))
    .resolves.toBe(accepted);
  expect(api.regenerateMessage).toHaveBeenLastCalledWith(
    "message-one", { temperature: 0.25 }, firstKey,
  );
  expect(localStorage.length).toBe(0);
});

it("treats reordered object fields as the same request but preserves array order", async () => {
  vi.mocked(api.regenerateMessage).mockRejectedValue(new Error("Queue full"));
  const send = (settings: Record<string, unknown>) =>
    regenerateWithRetry("chat-one", "message-one", settings).catch(() => undefined);
  await send({ nested: { alpha: 1, beta: 2 }, values: [1, 2] });
  await send({ values: [1, 2], nested: { beta: 2, alpha: 1 } });
  await send({ nested: { alpha: 1, beta: 2 }, values: [2, 1] });
  const calls = vi.mocked(api.regenerateMessage).mock.calls;
  expect(calls[1][2]).toBe(calls[0][2]);
  expect(calls[2][2]).not.toBe(calls[0][2]);
});

it("keeps independent settings and targets separate without forgetting earlier retries", async () => {
  vi.mocked(api.regenerateMessage).mockRejectedValue(new Error("Connection lost"));
  const cases: [string, string, Record<string, unknown>][] = [
    ["chat-one", "message-one", { temperature: 0.25 }],
    ["chat-one", "message-one", { temperature: 0.75 }],
    ["chat-one", "message-two", { temperature: 0.25 }],
    ["chat-two", "message-one", { temperature: 0.25 }],
    ["chat-one", "message-one", { temperature: 0.25 }],
  ];
  for (const args of cases) await regenerateWithRetry(...args).catch(() => undefined);
  const keys = vi.mocked(api.regenerateMessage).mock.calls.map((call) => call[2]);
  expect(new Set(keys.slice(0, 4)).size).toBe(4);
  expect(keys[4]).toBe(keys[0]);
});

it("gives the next deliberate regeneration a new key after acceptance", async () => {
  vi.mocked(api.regenerateMessage).mockResolvedValue(accepted);
  await regenerateWithRetry("chat-one", "message-one", {});
  await regenerateWithRetry("chat-one", "message-one", {});
  const calls = vi.mocked(api.regenerateMessage).mock.calls;
  expect(calls[1][2]).not.toBe(calls[0][2]);
});

it("does not send when the retry identity cannot be persisted", async () => {
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("Quota"); });
  await expect(regenerateWithRetry("chat-one", "message-one", {}))
    .rejects.toThrow("Check browser storage");
  expect(api.regenerateMessage).not.toHaveBeenCalled();
});

it("freezes the submitted settings before the digest yields", async () => {
  vi.mocked(api.regenerateMessage).mockResolvedValue(accepted);
  const settings = { nested: { temperature: 0.25 } };
  const pending = regenerateWithRetry("chat-one", "message-one", settings);
  settings.nested.temperature = 0.75;
  await pending;
  expect(api.regenerateMessage).toHaveBeenCalledWith(
    "message-one", { nested: { temperature: 0.25 } }, expect.any(String),
  );
});

it("does not turn accepted work into an error when clearing local storage fails", async () => {
  vi.mocked(api.regenerateMessage).mockResolvedValue(accepted);
  vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => { throw new Error("Storage"); });
  await expect(regenerateWithRetry("chat-one", "message-one", {})).resolves.toBe(accepted);
  expect(localStorage.length).toBe(1);
});
