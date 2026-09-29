import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { PropsWithChildren } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { buildTurnRequest } from "./turnRequest";
import type { SourceFitPreviewResult, SourceFitSelection } from "./sourceFit";
import { useSourceFitCanvas } from "./useSourceFitCanvas";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const sourceId = "sha256:" + "a".repeat(64);
const value: SourceFitSelection = {
  sourceArtifactId: sourceId,
  workflowRevisionId: "revision-one",
  request: { mode: "extend", width: 1200, height: 900 },
};
const answer: SourceFitPreviewResult = {
  version: 1, mode: "extend", workflow_revision_id: "revision-one",
  workflow_artifact_sha256: "b".repeat(64), source_artifact_id: sourceId,
  source: { width: 800, height: 600 }, canvas: { width: 1200, height: 900 },
  margins: { left: 200, right: 200, top: 150, bottom: 150 },
  source_rectangle: { x: 200, y: 150, width: 800, height: 600 },
  request_authorized: false,
};

function harness() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: PropsWithChildren) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  vi.spyOn(api, "workflowRevisionSourceFit").mockResolvedValue({
    available: true, reason: null, modes: ["extend"], request_authorized: false,
  });
  const legacy = vi.spyOn(api, "previewWorkflowRevisionSourceFit").mockResolvedValue(answer);
  const turn = vi.spyOn(api, "previewTurnSourceFit").mockResolvedValue(answer);
  const prior = vi.spyOn(api, "previewPriorTurnSourceFit").mockResolvedValue(answer);
  const props = {
    mode: "image" as const, sourceCanvasRevisionId: "revision-one", sourceId,
    families: [], workflowSelection: null, projectSelection: null, value,
    onChange: vi.fn(),
  };
  const request = buildTurnRequest({
    text: "Extend the neutral diagram", mode: "image", inputArtifactIds: [sourceId],
    settings: { steps: 12 }, references: [], outputCount: 2, sourceFit: value.request,
  });
  return { wrapper, legacy, turn, prior, props, request };
}

it.each(["turn", "prior-edit"] as const)("previews the complete %s request through its contextual endpoint", async (kind) => {
  const setup = harness();
  const request = { ...setup.request, idempotency_key: "request-one" };
  const previewContext = { kind, id: kind === "turn" ? "chat-one" : "message-one", request };
  const { result } = renderHook(() => useSourceFitCanvas({ ...setup.props, ...{ previewContext } }), { wrapper: setup.wrapper });
  await waitFor(() => expect(result.current.canPreview).toBe(true));
  act(() => result.current.requestPreview());
  const expected = kind === "turn" ? setup.turn : setup.prior;
  await waitFor(() => expect(expected).toHaveBeenCalledTimes(1));
  expect(expected).toHaveBeenCalledWith(previewContext.id, request, expect.any(AbortSignal));
  expect(setup.legacy).not.toHaveBeenCalled();
  await waitFor(() => expect(result.current.selection).toEqual(value));
});

it.each(["text", "settings", "references", "target"])("discards a preview when the submission %s changes", async (changed) => {
  const setup = harness();
  const previewContext = { kind: "turn" as const, id: "chat-one", request: setup.request };
  const { result, rerender } = renderHook((props) => useSourceFitCanvas(props), {
    initialProps: { ...setup.props, previewContext }, wrapper: setup.wrapper,
  });
  await waitFor(() => expect(result.current.canPreview).toBe(true));
  act(() => result.current.requestPreview());
  await waitFor(() => expect(result.current.preview).toEqual(answer));
  const request = {
    ...setup.request,
    ...(changed === "text" ? { text: "Extend the revised diagram" } : {}),
    ...(changed === "settings" ? { settings: { steps: 24 } } : {}),
    ...(changed === "references" ? { references: [{ reference_subject_id: "reference-two", source: "mention" as const }] } : {}),
  };
  rerender({ ...setup.props, previewContext: { ...previewContext, id: changed === "target" ? "chat-two" : previewContext.id, request } });
  expect(result.current.preview).toBeNull();
  expect(result.current.selection).toBeNull();
  expect(result.current.pending).toBe(false);
});

it("resolves an implicit source and workflow from the complete turn request", async () => {
  const setup = harness();
  const previewContext = {
    kind: "turn" as const, id: "chat-one",
    request: { ...setup.request, input_artifact_ids: [] },
  };
  const props = { ...setup.props, sourceId: null, sourceCanvasRevisionId: null,
    value: { ...value, sourceArtifactId: "", workflowRevisionId: "" }, previewContext };
  const { result } = renderHook(() => useSourceFitCanvas(props), { wrapper: setup.wrapper });
  expect(result.current.canPreview).toBe(true);
  act(() => result.current.requestPreview());
  await waitFor(() => expect(result.current.selection).toEqual(value));
  expect(setup.turn).toHaveBeenCalledWith("chat-one", previewContext.request, expect.any(AbortSignal));
  expect(api.workflowRevisionSourceFit).not.toHaveBeenCalled();
});

it("snapshots nested request values before the preview waits for its response", async () => {
  const setup = harness();
  const previewContext = { kind: "turn" as const, id: "chat-one", request: setup.request };
  const { result } = renderHook(() => useSourceFitCanvas({ ...setup.props, ...{ previewContext } }), { wrapper: setup.wrapper });
  await waitFor(() => expect(result.current.canPreview).toBe(true));
  act(() => result.current.requestPreview());
  await waitFor(() => expect(setup.turn).toHaveBeenCalledTimes(1));
  setup.request.settings.steps = 48;
  setup.request.input_artifact_ids.push("sha256:" + "c".repeat(64));
  expect(setup.turn.mock.calls[0][1].settings).toEqual({ steps: 12 });
  expect(setup.turn.mock.calls[0][1].input_artifact_ids).toEqual([sourceId]);
});

it("ignores a late answer after the preview target changes and requires another explicit preview", async () => {
  const setup = harness();
  let finish: (result: SourceFitPreviewResult) => void = () => {};
  setup.turn.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  const previewContext = { kind: "turn" as const, id: "chat-one", request: setup.request };
  const { result, rerender } = renderHook((props) => useSourceFitCanvas(props), {
    initialProps: { ...setup.props, previewContext }, wrapper: setup.wrapper,
  });
  await waitFor(() => expect(result.current.canPreview).toBe(true));
  act(() => result.current.requestPreview());
  await waitFor(() => expect(setup.turn).toHaveBeenCalledTimes(1));
  const signal = setup.turn.mock.calls[0][2];
  rerender({ ...setup.props, previewContext: { ...previewContext, id: "chat-two" } });
  expect(signal?.aborted).toBe(true);
  await act(async () => { finish(answer); });
  expect(result.current.preview).toBeNull();
  expect(result.current.selection).toBeNull();
  expect(setup.turn).toHaveBeenCalledTimes(1);
  act(() => result.current.requestPreview());
  await waitFor(() => expect(setup.turn).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(result.current.selection).toEqual(value));
});
