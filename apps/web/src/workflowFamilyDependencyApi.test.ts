import { afterEach, expect, it, vi } from "vitest";
afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); sessionStorage.clear(); localStorage.clear(); });
it("requests dependency summaries only for the explicit Library query", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" })))
    .mockImplementation(async () => new Response("[]"));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await api.workflowFamilies(undefined, false, true);
  await api.workflowFamilies("image", true);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/workflow-families?include_dependencies=true");
  expect(fetchMock.mock.calls[2][0]).toBe("/api/workflow-families?selector_capability=image&include_archived=true");
});
