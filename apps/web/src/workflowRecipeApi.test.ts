import { afterEach, expect, it, vi } from "vitest";

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

it("sends recipe CRUD through the authenticated API and encodes recipe identifiers", async () => {
  const payload = { name: "Fine detail", use_case: "image_generation" as const,
    settings_json: { steps: 24 }, enabled: true, is_default: false };
  const record = { ...payload, id: "recipe/a", builtin: false };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "recipe-session" })))
    .mockResolvedValueOnce(new Response(JSON.stringify(record), { status: 201 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(record)))
    .mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await expect(api.createWorkflowUseCasePreset(payload)).resolves.toEqual(record);
  await expect(api.replaceWorkflowUseCasePreset(record.id, payload)).resolves.toEqual(record);
  await expect(api.deleteWorkflowUseCasePreset(record.id)).resolves.toBeUndefined();
  expect(fetchMock.mock.calls.slice(1).map(([url, options]) => [url, options.method])).toEqual([
    ["/api/workflow-use-case-presets", "POST"],
    ["/api/workflow-use-case-presets/recipe%2Fa", "PUT"],
    ["/api/workflow-use-case-presets/recipe%2Fa", "DELETE"],
  ]);
  for (const [, options] of fetchMock.mock.calls.slice(1)) {
    expect(options.headers.get("x-local-lm-csrf")).toBe("recipe-session");
    expect(options.credentials).toBe("same-origin");
  }
  expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual(payload);
});

it("keeps workspace, project and chat endpoints distinct with exact choice bodies", async () => {
  const fetchMock = vi.fn().mockImplementation(async (url: string) => new Response(JSON.stringify(
    url === "/api/session" ? { csrf_token: "session" } : { mode: "automatic" },
  )));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  const signal = new AbortController().signal;
  await api.workflowUseCasePresets("image_edit", 200, signal);
  await api.workflowUseCaseDefault("image_edit", signal);
  await api.setWorkflowUseCaseDefault("image_edit", { preset_id: null });
  await api.workflowUseCaseChoice({ kind: "project", id: "p/a" }, "image_edit", signal);
  await api.setWorkflowUseCaseChoice({ kind: "project", id: "p/a" }, "image_edit", { mode: "inherit" });
  await api.workflowUseCaseChoice({ kind: "chat", id: "c/a" }, "image_edit", signal);
  await api.setWorkflowUseCaseChoice({ kind: "chat", id: "c/a" }, "image_edit", { mode: "preset", preset_id: "fine" });
  expect(fetchMock.mock.calls.slice(1).map(([url]) => url)).toEqual([
    "/api/workflow-use-case-presets?limit=200&offset=200&use_case=image_edit",
    "/api/workflow-use-case-defaults/image_edit", "/api/workflow-use-case-defaults/image_edit",
    "/api/projects/p%2Fa/workflow-use-case-presets/image_edit", "/api/projects/p%2Fa/workflow-use-case-presets/image_edit",
    "/api/chats/c%2Fa/workflow-use-case-presets/image_edit", "/api/chats/c%2Fa/workflow-use-case-presets/image_edit",
  ]);
  for (const index of [1, 2, 4, 6]) expect(fetchMock.mock.calls[index][1].signal).toBe(signal);
  for (const [index, value] of [[3, { preset_id: null }], [5, { mode: "inherit" }], [7, { mode: "preset", preset_id: "fine" }]] as const) {
    const options = fetchMock.mock.calls[index][1];
    expect(options.method).toBe("PUT");
    expect(JSON.parse(options.body)).toEqual(value);
    expect(options.headers.get("x-local-lm-csrf")).toBe("session");
  }
});
