import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals(); vi.resetModules(); sessionStorage.clear(); localStorage.clear();
});

it("sends bounded installed model filters and exact identities", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await expect(api.modelsPage({
    limit: 50, offset: 100, search: "Straße %_", role: "chat", chatCapability: "vision", modelIds: ["one/a", "two&b"],
  })).resolves.toEqual([]);
  const url = new URL(String(fetchMock.mock.calls[1][0]), "http://localhost");
  expect(url.pathname).toBe("/api/models");
  expect(url.searchParams.get("limit")).toBe("50");
  expect(url.searchParams.get("offset")).toBe("100");
  expect(url.searchParams.get("search")).toBe("Straße %_");
  expect(url.searchParams.get("role")).toBe("chat");
  expect(url.searchParams.get("chat_capability")).toBe("vision");
  expect(url.searchParams.getAll("model_id")).toEqual(["one/a", "two&b"]);
});

it("checks only requested catalog identities without collapsing workflow variants", async () => {
  const matches = { remote_ids: ["neutral/a&b"], workflow_template_ids: ["edit/one"] };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(matches), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await expect(api.catalogInstallMatches({
    role: "image", remoteIds: ["neutral/a&b", "neutral/two"], workflowTemplateIds: ["edit/one", "create/two"],
  })).resolves.toEqual(matches);
  const url = new URL(String(fetchMock.mock.calls[1][0]), "http://localhost");
  expect(url.pathname).toBe("/api/models/catalog-matches");
  expect(url.searchParams.get("role")).toBe("image");
  expect(url.searchParams.getAll("remote_id")).toEqual(["neutral/a&b", "neutral/two"]);
  expect(url.searchParams.getAll("workflow_template_id")).toEqual(["edit/one", "create/two"]);
});
