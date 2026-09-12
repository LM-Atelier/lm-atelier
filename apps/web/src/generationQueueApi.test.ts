import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals(); vi.resetModules(); sessionStorage.clear(); localStorage.clear();
});

const policy = {
  lane: "generation", dispatch_state: "paused", revision: 4,
  running_jobs: 0, allowed_actions: ["resume"],
};

it("reads generation state with cancellation and no body", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(policy), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  expect(api.generationQueuePolicy).toBeTypeOf("function");
  const controller = new AbortController();
  await expect(api.generationQueuePolicy(controller.signal)).resolves.toEqual(policy);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/queue/lanes/generation");
  expect(fetchMock.mock.calls[1][1]?.signal).toBe(controller.signal);
  expect(fetchMock.mock.calls[1][1]?.body).toBeUndefined();
});

it("posts each supported action with the exact revision and retry key", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(policy), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(policy), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  expect(api.generationQueueControl).toBeTypeOf("function");
  const command = { expected_revision: 3, idempotency_key: "neutral-retry-key" };
  await api.generationQueueControl("pause_after_current", command);
  await api.generationQueueControl("resume", command);
  for (const [index, path] of [[1, "pause-after-current"], [2, "resume"]] as const) {
    expect(fetchMock.mock.calls[index][0]).toBe("/api/queue/lanes/generation/" + path);
    expect(fetchMock.mock.calls[index][1]?.method).toBe("POST");
    expect(JSON.parse(fetchMock.mock.calls[index][1]?.body as string)).toEqual(command);
  }
});
