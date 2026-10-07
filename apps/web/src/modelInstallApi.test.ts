import { afterEach, expect, it, vi } from "vitest";
import type { ModelInstall } from "./types";

afterEach(() => {
  vi.unstubAllGlobals(); vi.resetModules(); sessionStorage.clear(); localStorage.clear();
});
function model(id: string): ModelInstall {
  return { id, source_id: null, name: "Granite checkpoint", role: "image", engine: "mock",
    local_path: "managed/granite", size_bytes: 10, compatibility: "likely", manifest_json: {},
    active: true, readiness: "unverified", capability_evidence: null,
    created_at: "2026-09-10T00:00:00Z", updated_at: "2026-09-10T00:00:00Z" };
}
it("requests one exact install and keeps the legacy list request unchanged", async () => {
  const selected = model("checkpoint/a&b");
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify([model("other"), selected]), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await expect(api.modelInstall(selected.id)).resolves.toEqual(selected);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/models?install_id=checkpoint%2Fa%26b");
  expect(fetchMock.mock.calls[1][1]?.body).toBeUndefined();
  await expect(api.models()).resolves.toEqual([]);
  expect(fetchMock.mock.calls[2][0]).toBe("/api/models");
});
it("does not substitute another installed model when the exact identity is absent", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify([model("other")]), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await expect(api.modelInstall("missing")).resolves.toBeNull();
});
