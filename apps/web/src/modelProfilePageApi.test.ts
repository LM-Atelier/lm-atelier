import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals(); vi.resetModules(); sessionStorage.clear(); localStorage.clear();
});

it("sends bounded preset searches and exact selected or default identities", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await expect(api.presetsPage({
    limit: 50, offset: 100, search: "Straße %_", role: "image", presetIds: ["one/a", "two&b"], defaultsOnly: true,
  })).resolves.toEqual([]);
  const url = new URL(String(fetchMock.mock.calls[1][0]), "http://localhost");
  expect(url.pathname).toBe("/api/presets");
  expect(url.searchParams.get("limit")).toBe("50");
  expect(url.searchParams.get("offset")).toBe("100");
  expect(url.searchParams.get("search")).toBe("Straße %_");
  expect(url.searchParams.get("role")).toBe("image");
  expect(url.searchParams.get("defaults_only")).toBe("true");
  expect(url.searchParams.getAll("preset_id")).toEqual(["one/a", "two&b"]);
});

it("sends profile pages and exact identities without treating them as query syntax", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await expect(api.profilesPage({
    limit: 50, offset: 50, search: "Straße %_", role: "image", engine: "comfyui",
    installIds: ["install/a&b", "another"], profileIds: ["selected/one", "selected&two"], defaultsOnly: false,
  })).resolves.toEqual([]);
  const url = new URL(String(fetchMock.mock.calls[1][0]), "http://localhost");
  expect(url.pathname).toBe("/api/profiles");
  expect(url.searchParams.get("limit")).toBe("50");
  expect(url.searchParams.get("offset")).toBe("50");
  expect(url.searchParams.get("search")).toBe("Straße %_");
  expect(url.searchParams.get("role")).toBe("image");
  expect(url.searchParams.get("engine")).toBe("comfyui");
  expect(url.searchParams.get("defaults_only")).toBe("false");
  expect(url.searchParams.getAll("install_id")).toEqual(["install/a&b", "another"]);
  expect(url.searchParams.getAll("profile_id")).toEqual(["selected/one", "selected&two"]);
  expect(fetchMock.mock.calls[1][1]?.body).toBeUndefined();
});

it("filters profile input capability before requesting a page", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await api.profilesPage({ role: "chat", inputModality: "image", limit: 50, offset: 100 });
  const url = new URL(String(fetchMock.mock.calls[1][0]), "http://localhost");
  expect(url.searchParams.get("input_modality")).toBe("image");
  expect(url.searchParams.get("role")).toBe("chat");
  expect(url.searchParams.get("limit")).toBe("50");
  expect(url.searchParams.get("offset")).toBe("100");
});
