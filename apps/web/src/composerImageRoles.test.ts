import { afterEach, expect, it, vi } from "vitest";
import { attachmentImageRoles, editSourceId, withImagePurpose } from "./composerImageRoles";
import { composerDraftFrom, storedDraft } from "./composerDraftStore";
import { buildPriorTurnEditRequest, initializePriorTurnEditDraft } from "./priorTurnEditDraft";
import { buildTurnRequest } from "./turnRequest";
import { turnPreviewContext } from "./useTurnEditorSourceFit";
import type { Artifact, ChatComposerDraft, ImageInputRole, PriorTurnEditSource } from "./types";
import type { ComposerAttachment } from "./useComposerUploads";

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); localStorage.clear(); sessionStorage.clear(); });

const pictures = (): ComposerAttachment[] => ["layout", "canvas", "detail"].map((id) => ({ id, kind: "image", origin: "uploaded" }));
const purposes: ImageInputRole[] = ["reference", "edit_source", "reference"];
function source(): PriorTurnEditSource {
  return {
    source_user_message_id: "message", source_run_id: "run", source_snapshot_sha256: "a".repeat(64),
    chat_id: "chat", text: "A garden", mode: "image", operation: "image_to_image",
    input_artifact_ids: pictures().map((item) => item.id), input_image_roles: [...purposes],
    input_artifacts: pictures().map((item): Artifact => ({ id: item.id, sha256: "b".repeat(64), kind: "input",
      media_type: "image/png", size_bytes: 1, original_name: null, metadata_json: {}, created_at: "2026-10-05T12:00:00Z" })),
    references: [], settings: {}, resolved_settings: {}, settings_role: "image", output_count: 1,
    profile_id: null, vision_profile_id: null, preset_id: null, preset: null, model_selection: {},
    workflow_selection: { selector_capability: "image", mode: "revision", workflow_revision_id: "workflow", workflow_family_id: null, legacy_profile_id: null },
    workflow_revision_id: "workflow", workflow_schema: {}, context_messages: [], prompt_source: null,
    source_fit: { mode: "crop", width: 512, height: 512 },
  };
}

it("selects a canvas in place and preserves both references in their chosen order", () => {
  const selected = withImagePurpose(pictures(), "canvas", "edit_source");
  expect(selected.map((item) => item.id)).toEqual(["layout", "canvas", "detail"]);
  expect(attachmentImageRoles(selected)).toEqual(purposes);
  expect(editSourceId(selected)).toBe("canvas");
  expect(attachmentImageRoles(withImagePurpose(selected, "detail", "edit_source"))).toEqual(["reference", "reference", "edit_source"]);
  expect(editSourceId(withImagePurpose(selected, "canvas", "reference"))).toBeNull();
  expect(attachmentImageRoles(withImagePurpose(selected, "canvas", undefined))).toBeUndefined();
  expect(editSourceId(pictures())).toBe("layout");
});

it("refuses partial or mixed purposes without crashing the canvas preview", () => {
  const selected = withImagePurpose(pictures(), "canvas", "edit_source");
  const mixed: ComposerAttachment[] = [...selected, { id: "clip", kind: "video", origin: "uploaded" }];
  expect(() => attachmentImageRoles(mixed)).toThrow("Choose a purpose");
  expect(turnPreviewContext("chat", "A garden", "image", mixed,
    { fields: [], chosenSettings: {}, settings: {}, references: [], requestedOutputCount: 1, promptSource: undefined }, undefined)).toBeUndefined();
});

it("persists unsent purposes and restores the exact attachment order", () => {
  const input = storedDraft({ text: "A garden", promptSource: null,
    editor: { requestId: "draft", mode: "image", attachments: withImagePurpose(pictures(), "canvas", "edit_source"),
      attachmentIntent: "replace", referenceIntent: "inherit", mentions: [], outputCount: 1, templateSettings: null } });
  const saved: ChatComposerDraft = { ...input, chat_id: "chat", revision: 1, updated_at: "2026-10-05T12:00:00Z" };
  const restored = composerDraftFrom(saved);
  expect(restored.editor?.attachments.map((item) => item.id)).toEqual(["layout", "canvas", "detail"]);
  expect(attachmentImageRoles(restored.editor?.attachments ?? [])).toEqual(purposes);
  expect(storedDraft(restored)).toEqual(input);
});

it("inherits purposes for a saved edit and binds its canvas to the second picture", () => {
  const draft = initializePriorTurnEditDraft(source());
  expect(draft.editor.sourceFit?.sourceArtifactId).toBe("canvas");
  expect(attachmentImageRoles(draft.editor.attachments)).toEqual(purposes);
  expect(buildPriorTurnEditRequest(draft).input_image_roles).toBeUndefined();
  draft.editor.attachmentIntent = "replace";
  expect(buildPriorTurnEditRequest(draft).input_image_roles).toEqual(purposes);
  draft.editor.attachments = withImagePurpose(draft.editor.attachments, "canvas", undefined);
  draft.editor.sourceFit = null;
  expect(buildPriorTurnEditRequest(draft).input_image_roles).toBeNull();
});

it("snapshots purposes and fits the selected canvas without moving its references", () => {
  const roles = [...purposes];
  const payload = buildTurnRequest({ text: "A garden", mode: "image", inputArtifactIds: ["layout", "canvas", "detail"],
    inputImageRoles: roles, settings: {}, sourceFit: { sourceArtifactId: "canvas", workflowRevisionId: "workflow",
      request: { mode: "crop", width: 512, height: 512 } } });
  roles[1] = "reference";
  expect(payload.input_image_roles).toEqual(purposes);
  expect(payload.input_artifact_ids).toEqual(["layout", "canvas", "detail"]);
  expect(() => buildTurnRequest({ text: "A garden", mode: "image", inputArtifactIds: ["canvas"],
    inputImageRoles: purposes, settings: {} })).toThrow("Choose a purpose");
});

it.each([false, true])("sends explicit purposes through the actual API transport (stop: %s)", async (stop) => {
  const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" })))
    .mockResolvedValueOnce(new Response(JSON.stringify({ work_plan_id: "plan" })));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  if (stop) await api.stopAndSendTurn("chat", "A garden", "image", ["layout", "canvas", "detail"], {}, "key", [], 1, undefined, undefined, undefined, purposes);
  else await api.sendTurn("chat", "A garden", "image", ["layout", "canvas", "detail"], {}, "key", "turns", undefined, [], 1, undefined, undefined, undefined, undefined, purposes);
  expect(fetchMock.mock.calls[1][0]).toBe(`/api/chats/chat/${stop ? "stop-and-send" : "turns"}`);
  expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toMatchObject({ input_artifact_ids: ["layout", "canvas", "detail"], input_image_roles: purposes });
});
