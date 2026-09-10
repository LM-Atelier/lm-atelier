import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals(); vi.resetModules(); sessionStorage.clear(); localStorage.clear();
});

it("encodes the queue cursor and forwards cancellation without a request body", async () => {
  const result = { items: [], total: 0, lane_counts: { generation: 0, transfer: 0, install: 0 },
    next_cursor: null, observed_at: "2026-09-01T00:00:00Z" };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(result), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(result), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  const controller = new AbortController();
  await expect(api.queueActivity({ lane: "transfer", cursor: "page+&=", limit: 25 }, controller.signal))
    .resolves.toEqual(result);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/queue/activity?limit=25&lane=transfer&cursor=page%2B%26%3D");
  expect(fetchMock.mock.calls[1][1]?.signal).toBe(controller.signal);
  expect(fetchMock.mock.calls[1][1]?.body).toBeUndefined();
  await api.queueActivity({ cursor: null, limit: 50 });
  expect(fetchMock.mock.calls[2][0]).toBe("/api/queue/activity?limit=50");
});

it("bounds step detail reads, encodes plan identity and refuses another plan's response", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify({ plan_id: "plan a&b", items: [], total: 0,
      next_offset: null, observed_at: "2026-09-01T00:00:00Z" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify({ plan_id: "other", items: [], total: 0,
      next_offset: null, observed_at: "2026-09-01T00:00:00Z" }), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  const controller = new AbortController();
  await api.queuePlanSteps("plan a&b", 50, controller.signal);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/queue/plans/plan%20a%26b/steps?limit=50&offset=50");
  expect(fetchMock.mock.calls[1][1]?.signal).toBe(controller.signal);
  await expect(api.queuePlanSteps("plan a&b", 0)).rejects.toThrow("The submitted work steps could not be read.");
});
