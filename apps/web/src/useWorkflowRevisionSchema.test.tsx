import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import type { WorkflowRevisionSchema } from "./types";
import { useWorkflowRevisionSchema } from "./useWorkflowRevisionSchema";

vi.mock("./api", () => ({ api: { workflowRevisionSchema: vi.fn() } }));
const clients: QueryClient[] = [];
afterEach(() => {
  cleanup();
  clients.forEach((client) => client.clear());
  clients.length = 0;
  vi.resetAllMocks();
});

function harness() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  return { client, wrapper: ({ children }: { children: ReactNode }) =>
    <QueryClientProvider client={client}>{children}</QueryClientProvider> };
}
function schema(id: string, operation = "text_to_image"): WorkflowRevisionSchema {
  return { revision_id: id, workflow_id: "example", operation,
    input_schema_json: { properties: { quality: { type: "number", default: id.length } } } };
}

it("makes no schema request for an unresolved revision", () => {
  const { wrapper } = harness();
  const { result } = renderHook(() => useWorkflowRevisionSchema(null, "text_to_image"), { wrapper });
  expect(result.current.schema).toBeUndefined();
  expect(api.workflowRevisionSchema).not.toHaveBeenCalled();
});

it("shares one exact revision request across composer and drawer consumers", async () => {
  const value = schema("selected");
  vi.mocked(api.workflowRevisionSchema).mockResolvedValue(value);
  const { wrapper } = harness();
  const { result } = renderHook(() => [
    useWorkflowRevisionSchema("selected", "text_to_image"),
    useWorkflowRevisionSchema("selected", "text_to_image"),
  ], { wrapper });
  await waitFor(() => expect(result.current.map((value) => value.schema)).toEqual([
    value.input_schema_json, value.input_schema_json,
  ]));
  expect(api.workflowRevisionSchema).toHaveBeenCalledTimes(1);
  expect(api.workflowRevisionSchema).toHaveBeenCalledWith("selected", expect.any(AbortSignal));
});

it("drops the old schema immediately when the selected revision changes", async () => {
  let finishSecond!: (value: WorkflowRevisionSchema) => void;
  vi.mocked(api.workflowRevisionSchema)
    .mockResolvedValueOnce(schema("first"))
    .mockImplementationOnce(() => new Promise((resolve) => { finishSecond = resolve; }));
  const { wrapper } = harness();
  const { result, rerender } = renderHook(
    ({ id }) => useWorkflowRevisionSchema(id, "text_to_image"),
    { wrapper, initialProps: { id: "first" } },
  );
  await waitFor(() => expect(result.current.schema).toEqual(schema("first").input_schema_json));
  rerender({ id: "second" });
  expect(result.current.schema).toBeUndefined();
  await act(async () => finishSecond(schema("second")));
  await waitFor(() => expect(result.current.schema).toEqual(schema("second").input_schema_json));
});

it("ignores a late response for a revision that is no longer selected", async () => {
  let finishFirst!: (value: WorkflowRevisionSchema) => void;
  vi.mocked(api.workflowRevisionSchema)
    .mockImplementationOnce(() => new Promise((resolve) => { finishFirst = resolve; }))
    .mockResolvedValueOnce(schema("second"));
  const { wrapper } = harness();
  const { result, rerender } = renderHook(
    ({ id }) => useWorkflowRevisionSchema(id, "text_to_image"),
    { wrapper, initialProps: { id: "first" } },
  );
  rerender({ id: "second" });
  await waitFor(() => expect(result.current.schema).toEqual(schema("second").input_schema_json));
  await act(async () => finishFirst(schema("first")));
  expect(result.current.schema).toEqual(schema("second").input_schema_json);
});

it("withholds cached settings after a failed refresh and restores them on retry", async () => {
  vi.mocked(api.workflowRevisionSchema)
    .mockResolvedValueOnce(schema("selected"))
    .mockRejectedValueOnce(new Error("Settings unavailable"))
    .mockResolvedValueOnce(schema("selected"));
  const { wrapper } = harness();
  const { result } = renderHook(
    () => useWorkflowRevisionSchema("selected", "text_to_image"), { wrapper },
  );
  await waitFor(() => expect(result.current.schema).toBeDefined());
  await act(async () => { await result.current.retry(); });
  await waitFor(() => expect(result.current.error?.message).toBe("Settings unavailable"));
  expect(result.current.schema).toBeUndefined();
  await act(async () => { await result.current.retry(); });
  await waitFor(() => expect(result.current.schema).toBeDefined());
  expect(result.current.error).toBeNull();
});

it.each([
  ["another", "text_to_image"],
  ["selected", "text_to_video"],
])("does not lend schema %s / %s to a different revision or operation", async (id, operation) => {
  vi.mocked(api.workflowRevisionSchema).mockResolvedValue(schema(id, operation));
  const queryHarness = harness();
  const { result } = renderHook(
    () => useWorkflowRevisionSchema("selected", "text_to_image"), { wrapper: queryHarness.wrapper },
  );
  await waitFor(() => expect(queryHarness.client.getQueryState(
    ["workflows", "revision-schema", "selected"],
  )?.status).toBe("success"));
  expect(result.current.schema).toBeUndefined();
});
