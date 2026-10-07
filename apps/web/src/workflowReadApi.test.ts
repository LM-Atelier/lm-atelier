import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
  sessionStorage.clear();
  localStorage.clear();
});

function transport(value: unknown) {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" })))
    .mockResolvedValueOnce(new Response(JSON.stringify(value)));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

it("pages ready revision choices with exact saved revisions and cancellation", async () => {
  const fetchMock = transport([]);
  const { api } = await import("./api");
  const signal = new AbortController().signal;
  await api.workflowReadyRevisions({ limit: 50, offset: 100, search: "Paper %",
    operation: "text_to_image", revisionIds: ["revision / one", "revision two"] }, signal);
  const url = new URL(fetchMock.mock.calls[1][0], "http://localhost");
  expect(url.pathname).toBe("/api/workflow-ready-revisions");
  expect(Object.fromEntries(url.searchParams)).toMatchObject({ limit: "50", offset: "100",
    search: "Paper %", operation: "text_to_image" });
  expect(url.searchParams.getAll("revision_id")).toEqual(["revision / one", "revision two"]);
  expect(fetchMock.mock.calls[1][1].signal).toBe(signal);
});

it("reads workflow summaries without requesting the bulk graph list", async () => {
  const fetchMock = transport([]);
  const { api } = await import("./api");
  expect(await api.workflowSummaries()).toEqual([]);
  expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
    "/api/session", "/api/workflow-summaries",
  ]);
});

it("requests only the explicitly selected workflow detail", async () => {
  const fetchMock = transport({ id: "workflow / one", revisions: [] });
  const { api } = await import("./api");
  expect(await api.workflow("workflow / one")).toEqual({ id: "workflow / one", revisions: [] });
  expect(fetchMock.mock.calls[1][0]).toBe("/api/workflows/workflow%20%2F%20one");
});

it("reads historical revision choices without requesting graphs", async () => {
  const choices = [{ revision_id: "old", workflow_id: "one", workflow_name: "Example",
    operation: "text_to_image", version: 1 }];
  const fetchMock = transport(choices);
  const { api } = await import("./api");
  expect(await api.workflowRevisionChoices()).toEqual(choices);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/workflow-revision-choices");
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it("addresses a schema read to exactly one encoded revision", async () => {
  const schema = { revision_id: "revision / one", workflow_id: "one", operation: "text_to_image",
    input_schema_json: { properties: { quality: { type: "number" } } } };
  const fetchMock = transport(schema);
  const { api } = await import("./api");
  expect(await api.workflowRevisionSchema("revision / one")).toEqual(schema);
  expect(fetchMock.mock.calls[1][0]).toBe(
    "/api/workflow-revisions/revision%20%2F%20one/settings-schema",
  );
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it("bounds and filters workflow summaries while retaining exact selections", async () => {
  const fetchMock = transport([]);
  const { api } = await import("./api");
  const signal = new AbortController().signal;
  await api.workflowSummaries({ limit: 50, offset: 100, search: "Paper %", operation: "image_to_image", workflowIds: ["one / two", "three"] }, signal);
  const url = new URL(fetchMock.mock.calls[1][0], "http://localhost");
  expect(Object.fromEntries(url.searchParams)).toMatchObject({ limit: "50", offset: "100", search: "Paper %", operation: "image_to_image" });
  expect(url.searchParams.getAll("workflow_id")).toEqual(["one / two", "three"]);
  expect(fetchMock.mock.calls[1][1].signal).toBe(signal);
});

it("bounds revision history and resolves selected revisions independently", async () => {
  const fetchMock = transport([]);
  const { api } = await import("./api");
  const signal = new AbortController().signal;
  await api.workflowRevisionChoices(signal, { limit: 50, offset: 50, role: "image", search: "Paper", revisionIds: ["revision / one"] });
  const url = new URL(fetchMock.mock.calls[1][0], "http://localhost");
  expect(Object.fromEntries(url.searchParams)).toMatchObject({ limit: "50", offset: "50", role: "image", search: "Paper", revision_id: "revision / one" });
  expect(fetchMock.mock.calls[1][1].signal).toBe(signal);
});

it("bounds family and variant reads with global filters and exact choices", async () => {
  const fetchMock = transport([]);
  const { api } = await import("./api");
  const signal = new AbortController().signal;
  await api.workflowFamilies("image", true, true, {
    limit: 25, offset: 50, variantLimit: 10, variantOffset: 20, search: "Paper %",
    operation: "image_to_image", readiness: "ready", source: "workflow", order: "preference",
    enabledOnly: true, defaultsOnly: true, familyIds: ["one / two", "three"],
  }, signal);
  const url = new URL(fetchMock.mock.calls[1][0], "http://localhost");
  expect(Object.fromEntries(url.searchParams)).toMatchObject({
    selector_capability: "image", include_archived: "true", include_dependencies: "true",
    limit: "25", offset: "50", variant_limit: "10", variant_offset: "20", search: "Paper %",
    operation: "image_to_image", readiness: "ready", source: "workflow", order: "preference",
    enabled_only: "true", defaults_only: "true",
  });
  expect(url.searchParams.getAll("family_ids")).toEqual(["one / two", "three"]);
  expect(fetchMock.mock.calls[1][1].signal).toBe(signal);
});

it("pages variants of an exact encoded family", async () => {
  const fetchMock = transport({ id: "one / two" });
  const { api } = await import("./api");
  const signal = new AbortController().signal;
  await api.workflowFamily("one / two", { variantLimit: 25, variantOffset: 50 }, signal);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/workflow-families/one%20%2F%20two?variant_limit=25&variant_offset=50");
  expect(fetchMock.mock.calls[1][1].signal).toBe(signal);
});

it("reads global operation choices independently of the visible family page", async () => {
  const fetchMock = transport(["image_to_image", "text_to_video"]);
  const { api } = await import("./api");
  expect(await api.workflowFamilyOperations(true)).toEqual(["image_to_image", "text_to_video"]);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/workflow-family-operations?include_archived=true");
});
it("encodes exact family variants separately from ungrouped summary pages", async () => {
  const fetchMock = transport([]);
  const { api } = await import("./api");
  await api.workflowFamilies(undefined, true, false, {
    workflowIds: ["late / variant"], limit: 1, variantLimit: 1, variantCapability: "image",
  });
  const url = new URL(fetchMock.mock.calls[1][0], "http://localhost");
  expect(url.searchParams.getAll("workflow_ids")).toEqual(["late / variant"]);
  expect(url.searchParams.has("workflow_id")).toBe(false);
  expect(url.searchParams.get("variant_capability")).toBe("image");
  fetchMock.mockResolvedValueOnce(new Response(JSON.stringify([]), { status: 200 }));
  await api.workflowSummaries({ ungroupedOnly: true, limit: 20 });
  const summary = new URL(fetchMock.mock.calls[2][0], "http://localhost");
  expect(summary.searchParams.get("ungrouped_only")).toBe("true");
});
