import { afterEach, expect, it, vi } from "vitest";

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

it("requests bounded scalar summaries with paging, project search and cancellation", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  const controller = new AbortController();
  await expect(api.chatSummaries("project-one", true, "color study", {
    limit: 50, offset: 100, searchProjects: true, signal: controller.signal,
  })).resolves.toEqual([]);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/chats/summaries?include_archived=true&query=color+study&project_id=project-one&limit=50&offset=100&search_projects=true");
  expect(fetchMock.mock.calls[1][1].signal).toBe(controller.signal);
});
