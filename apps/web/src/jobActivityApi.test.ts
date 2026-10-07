import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals(); vi.resetModules(); sessionStorage.clear(); localStorage.clear();
});

it("requests the bounded activity projection without changing legacy history", async () => {
  const result = { active: [], active_count: 0, recent_issues: [] };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(result), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await expect(api.jobActivity(200)).resolves.toEqual(result);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/jobs/activity?active_limit=200");
  expect(fetchMock.mock.calls[1][1]?.body).toBeUndefined();
  await expect(api.jobs()).resolves.toEqual([]);
  expect(fetchMock.mock.calls[2][0]).toBe("/api/jobs");
});
