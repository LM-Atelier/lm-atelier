import { afterEach, expect, it, vi } from "vitest";
import type { RoutingMode } from "./types";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
  sessionStorage.clear();
  localStorage.clear();
});

function selection() {
  return {
    sourceArtifactId: "sha256:" + "a".repeat(64),
    workflowRevisionId: "revision-previewed",
    request: { mode: "extend" as const, width: 1200, height: 900 },
  };
}

function response(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json" } });
}

function replies() {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(response({ csrf_token: "csrf" }))
    .mockImplementation(async () => response({ accepted: true }, 202));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

it.each(["turns", "stop-and-send"])("submits the previewed source and exact revision through %s", async (endpoint) => {
  const fetchMock = replies();
  const { api } = await import("./api");
  const fit = selection();
  const inputs = [fit.sourceArtifactId, "sha256:" + "b".repeat(64)];
  const refs = [{ reference_subject_id: "reference-one", source: "mention" as const }];
  if (endpoint === "turns") {
    await api.sendTurn("chat-one", "Extend the landscape", "image", inputs, { steps: 12 },
      "request-one", "turns", undefined, refs, 2, undefined, undefined, fit);
  } else {
    await api.stopAndSendTurn("chat-one", "Extend the landscape", "image", inputs, { steps: 12 },
      "request-one", refs, 2, undefined, undefined, fit);
  }
  expect(fetchMock.mock.calls[1][0]).toBe("/api/chats/chat-one/" + endpoint);
  expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
    input_artifact_ids: inputs, workflow_revision_id: "revision-previewed",
    source_fit: { mode: "extend", width: 1200, height: 900 },
    settings: { steps: 12 }, references: refs, output_count: 2, idempotency_key: "request-one",
  });
});

it.each(["source", "revision", "mode"])("refuses a stale %s binding before sending", async (change) => {
  const fetchMock = replies();
  const { api } = await import("./api");
  const fit = selection();
  const inputs = [change === "source" ? "sha256:" + "c".repeat(64) : fit.sourceArtifactId];
  const mode: RoutingMode = change === "mode" ? "video" : "image";
  const revision = change === "revision" ? "revision-other" : undefined;
  await expect(api.sendTurn("chat-one", "Extend the landscape", mode, inputs, {},
    "request-one", "turns", revision, [], 1, undefined, undefined, fit))
    .rejects.toThrow("The source or workflow changed. Preview the canvas again before sending.");
  expect(fetchMock).not.toHaveBeenCalled();
});

it("keeps the exact source and canvas snapshot across a confirmation retry", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(response({ csrf_token: "csrf" }))
    .mockResolvedValueOnce(response({ detail: {
      code: "route_confirmation_required", plan: { operation: "image_to_image" },
      estimate: { estimated_intermediate_bytes: 2048 },
    } }, 409))
    .mockResolvedValueOnce(response({ accepted: true }, 202));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  const fit = selection();
  const inputs = [fit.sourceArtifactId];
  const confirm = vi.fn(async () => {
    fit.request.width = 2048;
    fit.workflowRevisionId = "revision-changed";
    inputs[0] = "sha256:" + "d".repeat(64);
    return true;
  });
  await api.sendTurn("chat-one", "Extend the landscape", "image", inputs, {},
    "request-one", "turns", undefined, [], 1, undefined, confirm, fit);
  expect(confirm).toHaveBeenCalledTimes(1);
  const first = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
  const retry = JSON.parse(String(fetchMock.mock.calls[2][1]?.body));
  expect(first.source_fit).toEqual({ mode: "extend", width: 1200, height: 900 });
  expect(retry).toEqual({ ...first, confirm_media: true });
});

it("lets the server route Auto while retaining the explicitly previewed image canvas", async () => {
  const fetchMock = replies();
  const { api } = await import("./api");
  const fit = selection();
  await api.sendTurn("chat-one", "Extend the landscape", "auto", [fit.sourceArtifactId], {},
    "request-auto", "turns", undefined, [], undefined, undefined, undefined, fit);
  expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
    mode: "auto", source_fit: { mode: "extend", width: 1200, height: 900 },
    workflow_revision_id: "revision-previewed", input_artifact_ids: [fit.sourceArtifactId],
  });
});

it.each(["image", "auto"] as const)("uses the same context preview and send body for %s", async (mode) => {
  const fetchMock = replies();
  const { api } = await import("./api");
  const { buildTurnRequest } = await import("./api");
  const fit = selection();
  const signal = new AbortController().signal;
  const payload = buildTurnRequest({
    text: "Extend the landscape", mode, inputArtifactIds: [fit.sourceArtifactId],
    settings: { steps: 12, nested: { strength: 0.4 } }, idempotencyKey: "request-context",
    references: [{ reference_subject_id: "reference-one", source: "mention" }],
    outputCount: 2, sourceFit: fit,
  });
  await api.previewTurnSourceFit("chat / one", payload, signal);
  const previewCall = fetchMock.mock.calls[1];
  expect(previewCall[0]).toBe("/api/chats/chat%20%2F%20one/source-fit/preview");
  expect(previewCall[1]?.signal).toBe(signal);
  expect(JSON.parse(String(previewCall[1]?.body))).toEqual({
    text: "Extend the landscape", mode, input_artifact_ids: [fit.sourceArtifactId],
    settings: { steps: 12, nested: { strength: 0.4 } }, idempotency_key: "request-context",
    references: [{ reference_subject_id: "reference-one", source: "mention" }],
    ...(mode === "image" ? { output_count: 2 } : {}),
    workflow_revision_id: "revision-previewed",
    source_fit: { mode: "extend", width: 1200, height: 900 }, confirm_media: false,
  });
  await api.sendTurn("chat / one", "Extend the landscape", mode, [fit.sourceArtifactId],
    { steps: 12, nested: { strength: 0.4 } }, "request-context", "turns", undefined,
    [{ reference_subject_id: "reference-one", source: "mention" }], 2, undefined, undefined, fit);
  expect(fetchMock.mock.calls[2][1]?.body).toBe(previewCall[1]?.body);
});

it("sends a prior-turn context preview without claiming an idempotent edit", async () => {
  const fetchMock = replies();
  const { api } = await import("./api");
  const signal = new AbortController().signal;
  const payload = {
    text: "Extend the landscape", idempotency_key: "draft-key", mode: "auto" as const,
    source_run_id: "accepted-run", source_snapshot_sha256: "a".repeat(64),
    settings: {}, role_overrides: { image: { settings: { steps: 7 }, preset_id: null } },
    source_fit: { mode: "extend" as const, width: 1200, height: 900 },
  };
  await api.previewPriorTurnSourceFit("message / one", payload, signal);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/messages/message%20%2F%20one/edits/source-fit/preview");
  expect(fetchMock.mock.calls[1][1]?.signal).toBe(signal);
  expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toEqual(payload);
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it("retains all context preview fields across a confirmation retry", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(response({ csrf_token: "csrf" }))
    .mockResolvedValueOnce(response({ detail: {
      code: "route_confirmation_required", plan: { operation: "image_to_image" },
      estimate: { estimated_intermediate_bytes: 2048 },
    } }, 409))
    .mockResolvedValueOnce(response({ accepted: true }, 202));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  const fit = selection();
  const settings = { nested: { strength: 0.4 } };
  const references = [{ reference_subject_id: "reference-one", source: "mention" as const }];
  await api.sendTurn("chat-one", "Extend the landscape", "image", [fit.sourceArtifactId], settings,
    "request-one", "turns", undefined, references, 1, undefined, async () => {
      settings.nested.strength = 0.9;
      references[0].reference_subject_id = "reference-changed";
      return true;
    }, fit);
  const first = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
  const retry = JSON.parse(String(fetchMock.mock.calls[2][1]?.body));
  expect(first.settings).toEqual({ nested: { strength: 0.4 } });
  expect(first.references).toEqual([{ reference_subject_id: "reference-one", source: "mention" }]);
  expect(retry).toEqual({ ...first, confirm_media: true });
});


it("leaves the implicit primary and workflow for the context preview server to select", async () => {
  const { buildTurnRequest } = await import("./api");
  const payload = buildTurnRequest({
    text: "Extend the landscape", mode: "auto", inputArtifactIds: [],
    settings: { steps: 3 }, sourceFit: { mode: "extend", width: 1200, height: 900 },
  });
  expect(payload).toEqual({
    text: "Extend the landscape", mode: "auto", input_artifact_ids: [], references: [],
    settings: { steps: 3 }, source_fit: { mode: "extend", width: 1200, height: 900 }, confirm_media: false,
  });
});
