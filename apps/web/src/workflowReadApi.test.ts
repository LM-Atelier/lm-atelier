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
