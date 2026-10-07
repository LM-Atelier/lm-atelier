import { afterEach, expect, it, vi } from "vitest";
afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); sessionStorage.clear(); localStorage.clear(); });

it("uses separate summary and exact detail reads while preserving the bulk client", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify({ id: "workflow a&b", revisions: [] }), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await api.workflowSummaries();
  const controller = new AbortController();
  await api.workflow("workflow a&b", controller.signal);
  await api.workflows();
  expect(fetchMock.mock.calls.slice(1).map(([path]) => path))
    .toEqual(["/api/workflow-summaries", "/api/workflows/workflow%20a%26b", "/api/workflows"]);
  expect(fetchMock.mock.calls[2][1]?.signal).toBe(controller.signal);
});

it("refuses a detail response for another workflow", async () => {
  vi.stubGlobal("fetch", vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify({ id: "other", revisions: [] }), { status: 200 })));
  const { api } = await import("./api");
  await expect(api.workflow("selected")).rejects.toThrow("The selected workflow could not be read.");
});
