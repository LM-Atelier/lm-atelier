import { expect, it, vi } from "vitest";
import * as draftApi from "./priorTurnEditDraft";
import {
  buildPriorTurnEditRequest, confirmPriorTurnEditSubmission, initializePriorTurnEditDraft, preparePriorTurnEditSubmission,
  priorTurnEditRequestFingerprint, readPriorTurnEditDraft, removePriorTurnEditDraft, writePriorTurnEditDraft,
  type PriorTurnEditStorage,
} from "./priorTurnEditDraft";
import type { PriorTurnEditSource } from "./types";

function memoryStorage(): PriorTurnEditStorage {
  const values = new Map<string, string>();
  return {
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => { values.set(key, value); }),
    removeItem: vi.fn((key: string) => { values.delete(key); }),
  };
}

function source(): PriorTurnEditSource {
  return {
    source_user_message_id: "source-user", source_run_id: "source-run", source_snapshot_sha256: "a".repeat(64),
    chat_id: "chat-one", text: "  Paint a green landscape with @ada  ", mode: "image", operation: "image_to_image",
    input_artifact_ids: ["image-one"],
    input_artifacts: [{ id: "image-one", sha256: "b".repeat(64), kind: "input", media_type: "image/png",
      size_bytes: 10, original_name: "landscape.png", metadata_json: {}, created_at: "2026-09-07T12:00:00Z" }],
    references: [{ reference_subject_id: "subject-one", mention_slug: "ada", subject_name: "Ada", subject_kind: "person",
      role: "subject", strength: 0.75, source: "mention", reference_asset_ids_json: ["asset-one"], artifact_ids_json: ["image-one"] }],
    settings: { width: 768 }, resolved_settings: { width: 768, height: 512 }, settings_role: "image", output_count: 3,
    profile_id: "profile-one", vision_profile_id: "vision-one", preset_id: "preset-one", preset: { name: "Landscape" },
    model_selection: {}, workflow_selection: { selector_capability: "image", mode: "revision", workflow_family_id: "family-one",
      workflow_revision_id: "revision-one", legacy_profile_id: null }, workflow_revision_id: "revision-one",
    workflow_schema: { type: "object" }, context_messages: [{ role: "user", content: "A green landscape." }], prompt_source: null,
  };
}

it("keeps Auto routing and role-specific settings through drawer navigation", () => {
  let draft = initializePriorTurnEditDraft({ ...source(), original_mode: "auto" });
  expect(draft.editor.mode).toBe("auto");
  draft.settings.width = 1024;
  draft.editor.templateSettings = { name: "Neutral template", settings: { seed: 42 } };
  draft = draftApi.selectPriorTurnEditConfiguration(draft, { role: "video" });
  draft.settings.fps = 12;
  draft.profileChoice = { kind: "explicit", value: "video-profile" };
  const before = buildPriorTurnEditRequest(draft);
  draft = draftApi.selectPriorTurnEditConfiguration(draft, { role: "image" });
  expect(draft.settings.width).toBe(1024);
  expect(draft.editor.templateSettings?.name).toBe("Neutral template");
  expect(buildPriorTurnEditRequest(draft)).toEqual(before);
  expect(before).toMatchObject({ mode: "auto", settings: { seed: 42 }, role_overrides: {
    image: { settings: { width: 1024 } }, video: { settings: { fps: 12 }, profile_id: "video-profile" },
  } });
});

it("keeps repeated source steps distinct and never seeds a role from the first step", () => {
  const first = source();
  first.original_mode = "auto";
  first.steps = [
    { ...first, step_id: "first", ordinal: 0, depends_on: [], settings: { steps: 5 } },
    { ...first, step_id: "second", source_run_id: "second-run", ordinal: 1, depends_on: ["first"], settings: { steps: 9 } },
  ];
  let draft = initializePriorTurnEditDraft(first);
  draft = draftApi.selectPriorTurnEditConfiguration(draft, { stepId: "first" });
  expect(draft.settings.steps).toBe(5);
  draft.settings.steps = 6;
  draft = draftApi.selectPriorTurnEditConfiguration(draft, { stepId: "second" });
  expect(draft.settings.steps).toBe(9);
  draft.settings.steps = 10;
  draft.presetChoice = { kind: "explicit", value: null };
  const before = buildPriorTurnEditRequest(draft);
  draft = draftApi.selectPriorTurnEditConfiguration(draft, { role: "image" });
  expect(draft.settings).toEqual({});
  expect(buildPriorTurnEditRequest(draft)).toEqual(before);
  expect(before.step_overrides).toEqual({ first: { settings: { steps: 6 } }, second: { settings: { steps: 10 }, preset_id: null } });
  expect(before.role_overrides).toBeUndefined();
  expect(() => draftApi.selectPriorTurnEditConfiguration(draft, { stepId: "missing" })).toThrow();
});

it("preserves a legacy confirmed wire and key through configuration migration and navigation", () => {
  const storage = memoryStorage();
  const draft = initializePriorTurnEditDraft(source());
  delete draft.configurations;
  draft.settingsRole = "video";
  draft.settings = { fps: 12 };
  draft.editor.mode = "auto";
  const editable = { text: draft.composer.text, idempotency_key: "legacy-key", source_run_id: draft.source.source_run_id,
    source_snapshot_sha256: draft.source.source_snapshot_sha256, mode: "auto" as const, settings: { fps: 12 }, output_count: 3 };
  const confirmed = { ...editable, mode: "video" as const, confirm_media: true };
  draft.pending = { request: confirmed, fingerprint: priorTurnEditRequestFingerprint(draft.source, confirmed),
    editableFingerprint: priorTurnEditRequestFingerprint(draft.source, editable) };
  writePriorTurnEditDraft(draft, storage);
  const restored = readPriorTurnEditDraft("chat-one", "source-user", storage)!;
  const navigated = draftApi.selectPriorTurnEditConfiguration(restored, { role: "image" });
  expect(preparePriorTurnEditSubmission(navigated, undefined, storage).request).toEqual(confirmed);
  navigated.composer.text = "A newer green landscape";
  expect(preparePriorTurnEditSubmission(navigated, undefined, storage).request.idempotency_key).not.toBe("legacy-key");
});

it("initializes an independent complete source draft with inherited selections and bindings", () => {
  const original = source();
  const draft = initializePriorTurnEditDraft(original);
  original.settings.width = 1024;
  original.input_artifacts[0].original_name = "changed.png";
  const request = buildPriorTurnEditRequest(draft);
  expect(draft.composer.text).toBe("  Paint a green landscape with @ada  ");
  expect(draft.editor).toMatchObject({ mode: "image", outputCount: 3, attachmentIntent: "inherit", referenceIntent: "inherit" });
  expect(draft.editor.attachments[0].artifact?.original_name).toBe("landscape.png");
  expect(request).toMatchObject({ settings: {}, source_run_id: "source-run", source_snapshot_sha256: "a".repeat(64) });
  for (const field of ["input_artifact_ids", "references", "preset_id", "workflow_selection", "profile_id", "vision_profile_id"]) {
    expect(request).not.toHaveProperty(field);
  }
});

it("distinguishes explicit clearing and same-ID reselection from inheritance in all choices", () => {
  const draft = initializePriorTurnEditDraft(source());
  const inherited = priorTurnEditRequestFingerprint(draft.source, buildPriorTurnEditRequest(draft));
  draft.presetChoice = { kind: "explicit", value: null };
  draft.profileChoice = { kind: "explicit", value: "profile-one" };
  draft.visionProfileChoice = { kind: "explicit", value: null };
  draft.workflowChoice = { kind: "explicit", value: { selector_capability: "image", mode: "revision", workflow_revision_id: "revision-one" } };
  const request = buildPriorTurnEditRequest(draft);
  expect(request).toMatchObject({ preset_id: null, profile_id: "profile-one", vision_profile_id: null,
    workflow_selection: { mode: "revision", workflow_revision_id: "revision-one" } });
  expect(priorTurnEditRequestFingerprint(draft.source, request)).not.toBe(inherited);
  draft.presetChoice = { kind: "explicit", value: "preset-one" };
  draft.workflowChoice = { kind: "explicit", value: null };
  expect(buildPriorTurnEditRequest(draft)).toMatchObject({ preset_id: "preset-one", workflow_selection: null });
});

it("preserves ordered replacement inputs and exact surviving reference options", () => {
  const draft = initializePriorTurnEditDraft(source());
  draft.editor.attachmentIntent = "replace";
  draft.editor.attachments = [{ id: "second", kind: "image", origin: "uploaded" }, ...draft.editor.attachments];
  draft.editor.referenceIntent = "replace";
  expect(buildPriorTurnEditRequest(draft)).toMatchObject({ input_artifact_ids: ["second", "image-one"], references: [{
    reference_subject_id: "subject-one", role: "subject", strength: 0.75, selected_asset_ids: ["asset-one"], source: "mention",
  }] });
  draft.editor.attachments = [];
  draft.composer.text = "A green landscape.";
  expect(buildPriorTurnEditRequest(draft)).toMatchObject({ input_artifact_ids: [], references: [] });
});

it("persists the complete pending request before a caller sends and restores its exact retry key", () => {
  const storage = memoryStorage();
  const first = preparePriorTurnEditSubmission(initializePriorTurnEditDraft(source()), undefined, storage);
  const restored = readPriorTurnEditDraft("chat-one", "source-user", storage)!;
  expect(restored.pending?.request).toEqual(first.request);
  const retried = preparePriorTurnEditSubmission(restored, undefined, storage);
  expect(retried.request).toEqual(first.request);
  expect(storage.setItem).toHaveBeenCalledTimes(2);
});

it("rotates identity when local intent or source binding changes despite identical displayed values", () => {
  const storage = memoryStorage();
  const first = preparePriorTurnEditSubmission(initializePriorTurnEditDraft(source()), undefined, storage);
  first.draft.presetChoice = { kind: "explicit", value: "preset-one" };
  const explicit = preparePriorTurnEditSubmission(first.draft, undefined, storage);
  expect(explicit.request.idempotency_key).not.toBe(first.request.idempotency_key);
  explicit.draft.source.source_snapshot_sha256 = "c".repeat(64);
  const changed = preparePriorTurnEditSubmission(explicit.draft, undefined, storage);
  expect(changed.request.idempotency_key).not.toBe(explicit.request.idempotency_key);
});

it("canonicalizes object fields while retaining array order and excluding request keys", () => {
  const draft = initializePriorTurnEditDraft(source());
  const first = buildPriorTurnEditRequest(draft);
  first.settings = { nested: { width: 768, height: 512 }, values: [1, 2] };
  const second = { ...first, idempotency_key: "different-key", settings: { values: [1, 2], nested: { height: 512, width: 768 } } };
  expect(priorTurnEditRequestFingerprint(draft.source, first)).toBe(priorTurnEditRequestFingerprint(draft.source, second));
  second.settings.values.reverse();
  expect(priorTurnEditRequestFingerprint(draft.source, first)).not.toBe(priorTurnEditRequestFingerprint(draft.source, second));
  expect(priorTurnEditRequestFingerprint({ ...draft.source, chat_id: "other-chat" }, first))
    .not.toBe(priorTurnEditRequestFingerprint(draft.source, first));
});

it("freezes nested request and draft values independently of later caller mutation", () => {
  const storage = memoryStorage();
  const draft = initializePriorTurnEditDraft(source());
  draft.settings = { nested: { width: 768 } };
  const prepared = preparePriorTurnEditSubmission(draft, undefined, storage);
  (draft.settings.nested as Record<string, unknown>).width = 1024;
  (prepared.request.settings!.nested as Record<string, unknown>).width = 1536;
  expect(prepared.draft.pending?.request.settings).toEqual({ nested: { width: 768 } });
  expect(readPriorTurnEditDraft("chat-one", "source-user", storage)?.settings).toEqual({ nested: { width: 768 } });
});

it("refuses preparation when persistence fails, leaving the original draft unchanged", () => {
  const storage = memoryStorage();
  vi.mocked(storage.setItem).mockImplementation(() => { throw new Error("Quota"); });
  const draft = initializePriorTurnEditDraft(source());
  expect(() => preparePriorTurnEditSubmission(draft, undefined, storage)).toThrow("Check browser storage");
  expect(draft.pending).toBeUndefined();
});

it("reports malformed saved data without deleting it or creating a replacement", () => {
  const storage = memoryStorage();
  vi.mocked(storage.getItem).mockReturnValue("{broken");
  expect(() => readPriorTurnEditDraft("chat-one", "source-user", storage)).toThrow("Could not restore");
  expect(storage.removeItem).not.toHaveBeenCalled();
  expect(storage.setItem).not.toHaveBeenCalled();
});

it("saves unsent drafts and removes only the accepted request, tolerating cleanup failures", () => {
  const storage = memoryStorage();
  const draft = initializePriorTurnEditDraft(source());
  writePriorTurnEditDraft(draft, storage);
  expect(readPriorTurnEditDraft("chat-one", "source-user", storage)).toEqual(draft);
  const first = preparePriorTurnEditSubmission(draft, undefined, storage);
  first.draft.composer.text = "A blue landscape.";
  const second = preparePriorTurnEditSubmission(first.draft, undefined, storage);
  removePriorTurnEditDraft("chat-one", "source-user", first.request.idempotency_key, storage);
  expect(readPriorTurnEditDraft("chat-one", "source-user", storage)?.pending?.request).toEqual(second.request);
  vi.mocked(storage.removeItem).mockImplementationOnce(() => { throw new Error("Unavailable"); });
  expect(() => removePriorTurnEditDraft("chat-one", "source-user", second.request.idempotency_key, storage)).not.toThrow();
  removePriorTurnEditDraft("chat-one", "source-user", second.request.idempotency_key, storage);
  expect(readPriorTurnEditDraft("chat-one", "source-user", storage)).toBeNull();
});

it("omits inherited settings but preserves changed values and deliberate same-value reselection", () => {
  const initial = source();
  initial.settings.loras = [{ id: "auxiliary-one", strength: 0.75 }];
  const draft = initializePriorTurnEditDraft(initial);
  expect(draft.settings.width).toBe(768);
  draft.settings.width = 1024;
  expect(buildPriorTurnEditRequest(draft).settings).toEqual({ width: 1024 });
  draft.explicitSettingsKeys = ["loras"];
  expect(buildPriorTurnEditRequest(draft).settings).toEqual({ width: 1024, loras: [{ id: "auxiliary-one", strength: 0.75 }] });
});
it("keeps picker and context bindings when deleting or adding mentions until explicitly removed", () => {
  const original = source();
  original.references.push({ reference_subject_id: "picked-subject", mention_slug: "picked", subject_name: "Picked subject",
    subject_kind: "object", source: "picker", role: "style", strength: 0.5,
    reference_asset_ids_json: ["picked-asset"], artifact_ids_json: ["picked-image"] });
  const draft = initializePriorTurnEditDraft(original);
  draft.composer.text = "A green landscape.";
  expect(buildPriorTurnEditRequest(draft).references).toEqual([{
    reference_subject_id: "picked-subject", source: "picker", role: "style", strength: 0.5, selected_asset_ids: ["picked-asset"],
  }]);
  draft.editor.mentions.push({ referenceSubjectId: "new-subject", mentionSlug: "new" });
  draft.composer.text += " @new";
  draft.editor.referenceIntent = "replace";
  expect(buildPriorTurnEditRequest(draft).references).toEqual([
    { reference_subject_id: "picked-subject", source: "picker", role: "style", strength: 0.5, selected_asset_ids: ["picked-asset"] },
    { reference_subject_id: "new-subject", source: "mention" },
  ]);
  draft.composer.text = "A green landscape.";
  draft.removedReferenceSubjectIds = ["picked-subject"];
  expect(buildPriorTurnEditRequest(draft).references).toEqual([]);
  draft.source.references[1].source = "inherited_context";
  draft.removedReferenceSubjectIds = [];
  expect(buildPriorTurnEditRequest(draft).references?.[0]).toMatchObject({ source: "inherited_context", selected_asset_ids: ["picked-asset"] });
});

it("explicitly removes a source reference even while all mention bindings are inherited", () => {
  const draft = initializePriorTurnEditDraft(source());
  draft.removedReferenceSubjectIds = ["subject-one"];
  expect(buildPriorTurnEditRequest(draft).references).toEqual([]);
});

const invalidDraftFields: [string[], unknown][] = [
  [["source", "settings"], null],
  [["source", "mode"], "unknown"],
  [["source", "references"], [{ reference_subject_id: "missing-binding" }]],
  [["editor", "mode"], "unknown"],
  [["editor", "attachmentIntent"], "unknown"],
  [["editor", "referenceIntent"], "unknown"],
  [["editor", "outputCount"], 0],
  [["editor", "outputCount"], 1.5],
  [["editor", "templateSettings"], { name: "Neutral settings", settings: [] }],
  [["presetChoice"], { kind: "explicit", value: 42 }],
  [["profileChoice"], { kind: "explicit", value: {} }],
  [["visionProfileChoice"], { kind: "explicit" }],
  [["workflowChoice"], { kind: "explicit", value: { selector_capability: "image", mode: "revision" } }],
  [["workflowChoice"], { kind: "explicit", value: { selector_capability: "unknown", mode: "default" } }],
  [["removedReferenceSubjectIds"], [42]],
  [["explicitSettingsKeys"], {}],
  [["pending"], null],
];

it.each(invalidDraftFields)("refuses malformed saved fields %j without replacing the draft", (path, replacement) => {
  const storage = memoryStorage();
  const value = JSON.parse(JSON.stringify(initializePriorTurnEditDraft(source()))) as Record<string, unknown>;
  let parent = value;
  for (const segment of path.slice(0, -1)) parent = parent[segment] as Record<string, unknown>;
  parent[path[path.length - 1]] = replacement;
  vi.mocked(storage.getItem).mockReturnValue(JSON.stringify(value));
  expect(() => readPriorTurnEditDraft("chat-one", "source-user", storage)).toThrow("Could not restore");
  expect(storage.setItem).not.toHaveBeenCalled();
  expect(storage.removeItem).not.toHaveBeenCalled();
});

it.each(["source_run_id", "source_snapshot_sha256"] as const)("refuses a mismatched pending %s even with a recomputed fingerprint", (field) => {
  const storage = memoryStorage();
  const prepared = preparePriorTurnEditSubmission(initializePriorTurnEditDraft(source()), undefined, storage);
  prepared.draft.pending!.request[field] = "different-source";
  prepared.draft.pending!.fingerprint = priorTurnEditRequestFingerprint(prepared.draft.source, prepared.draft.pending!.request);
  vi.mocked(storage.getItem).mockReturnValue(JSON.stringify(prepared.draft));
  expect(() => readPriorTurnEditDraft("chat-one", "source-user", storage)).toThrow("Could not restore");
});

it("refuses a pending request transplanted from another target", () => {
  const storage = memoryStorage();
  const prepared = preparePriorTurnEditSubmission(initializePriorTurnEditDraft(source()), undefined, storage);
  prepared.draft.source.source_user_message_id = "other-source-user";
  vi.mocked(storage.getItem).mockReturnValue(JSON.stringify(prepared.draft));
  expect(() => readPriorTurnEditDraft("chat-one", "other-source-user", storage)).toThrow("Could not restore");
});
it("retries the exact confirmed request after reopening without rotating its identity", () => {
  const storage = memoryStorage();
  const initial = initializePriorTurnEditDraft(source());
  initial.editor.mode = "auto";
  const first = preparePriorTurnEditSubmission(initial, undefined, storage);
  const request = { ...first.request, mode: "video" as const, confirm_media: true };
  first.draft.pending = {
    editableFingerprint: first.draft.pending!.fingerprint,
    fingerprint: priorTurnEditRequestFingerprint(first.draft.source, request), request,
  };
  writePriorTurnEditDraft(first.draft, storage);
  const restored = readPriorTurnEditDraft("chat-one", "source-user", storage)!;
  expect(preparePriorTurnEditSubmission(restored, undefined, storage).request).toEqual(request);
});

it("removes an accepted confirmed draft while preserving a newer local edit", () => {
  const storage = memoryStorage();
  const first = preparePriorTurnEditSubmission(initializePriorTurnEditDraft(source()), undefined, storage);
  const request = { ...first.request, confirm_media: true };
  first.draft.pending = {
    editableFingerprint: first.draft.pending!.fingerprint,
    fingerprint: priorTurnEditRequestFingerprint(first.draft.source, request), request,
  };
  writePriorTurnEditDraft(first.draft, storage);
  first.draft.composer.text = "A blue landscape.";
  writePriorTurnEditDraft(first.draft, storage);
  removePriorTurnEditDraft("chat-one", "source-user", request.idempotency_key, storage);
  expect(readPriorTurnEditDraft("chat-one", "source-user", storage)).not.toBeNull();
  first.draft.composer.text = source().text;
  writePriorTurnEditDraft(first.draft, storage);
  removePriorTurnEditDraft("chat-one", "source-user", request.idempotency_key, storage);
  expect(readPriorTurnEditDraft("chat-one", "source-user", storage)).toBeNull();
});


it("persists confirmation before returning and clears it only for a changed draft", () => {
  const storage = memoryStorage();
  const first = preparePriorTurnEditSubmission(initializePriorTurnEditDraft(source()), undefined, storage);
  const confirmed = confirmPriorTurnEditSubmission(first.draft, "video", storage);
  expect(readPriorTurnEditDraft("chat-one", "source-user", storage)?.pending?.request).toEqual(confirmed.request);
  expect(confirmed.request).toEqual({ ...first.request, mode: "video", confirm_media: true });
  expect(first.draft.pending?.request.confirm_media).toBeUndefined();
  confirmed.draft.composer.text = "A blue landscape.";
  const changed = preparePriorTurnEditSubmission(confirmed.draft, undefined, storage);
  expect(changed.request.confirm_media).toBeUndefined();
  expect(changed.request.idempotency_key).not.toBe(confirmed.request.idempotency_key);
});

it("refuses confirmation when storage fails without changing the pending request", () => {
  const storage = memoryStorage();
  const first = preparePriorTurnEditSubmission(initializePriorTurnEditDraft(source()), undefined, storage);
  vi.mocked(storage.setItem).mockImplementationOnce(() => { throw new Error("Quota"); });
  expect(() => confirmPriorTurnEditSubmission(first.draft, "video", storage)).toThrow("Check browser storage");
  expect(first.draft.pending?.request).toEqual(first.request);
});
