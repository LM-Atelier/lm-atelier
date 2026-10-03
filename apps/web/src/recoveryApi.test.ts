import { expect, it, vi } from "vitest";
import { recoveryApi } from "./recoveryApi";

it("materializes the bounded selection once and applies only its encoded batch identity and frozen command", async () => {
  const request = vi.fn().mockResolvedValue({});
  const api = recoveryApi(request);
  const signal = new AbortController().signal;
  const selection = { deletion_ids: ["garden", "travel"], action: "purge" as const, restore_unfiled: false };
  const command = { expected_revision: "a".repeat(64), impact_sha256: "b".repeat(64), operation_key: "remove-selection", acknowledgement: "permanently-delete" as const };
  await api.previewRecoveryBatch(selection, signal);
  expect(request).toHaveBeenLastCalledWith("/api/recovery-items/batches", { method: "POST", body: JSON.stringify(selection), signal });
  await api.applyRecoveryBatch("selection/garden", command);
  expect(request).toHaveBeenLastCalledWith("/api/recovery-items/batches/selection%2Fgarden/apply", { method: "POST", body: JSON.stringify(command) });
});

it("binds workflow family preview and Trash to one encoded identity and frozen command", async () => {
  const request = vi.fn().mockResolvedValue({});
  const api = recoveryApi(request);
  const signal = new AbortController().signal;
  const command = { expected_revision: "a".repeat(64), impact_sha256: "b".repeat(64), operation_key: "trash-workflow" };
  await api.workflowDeletionImpact("family/garden", signal);
  expect(request).toHaveBeenLastCalledWith("/api/workflow-families/family%2Fgarden/deletion-impact", { signal });
  await api.trashWorkflow("family/garden", command);
  expect(request).toHaveBeenLastCalledWith("/api/workflow-families/family%2Fgarden/trash", { method: "POST", body: JSON.stringify(command) });
});

it("binds Project preview and Trash to one encoded identity and frozen command", async () => {
  const request = vi.fn().mockResolvedValue({});
  const api = recoveryApi(request);
  const signal = new AbortController().signal;
  const command = { expected_revision: "a".repeat(64), impact_sha256: "b".repeat(64), operation_key: "trash-project" };
  await api.projectDeletionImpact("project/garden", signal);
  expect(request).toHaveBeenLastCalledWith("/api/projects/project%2Fgarden/deletion-impact", { signal });
  await api.trashProject("project/garden", command);
  expect(request).toHaveBeenLastCalledWith("/api/projects/project%2Fgarden/trash", { method: "POST", body: JSON.stringify(command) });
});

it("keeps filters, seek cursors and cancellation on the shared request path", async () => {
  const request = vi.fn().mockResolvedValue({ items: [], next_cursor: null });
  const api = recoveryApi(request);
  const signal = new AbortController().signal;
  await api.recoveryItems({ cursor: "next+/=", state: "blocked", kind: "media_library_entry", deletedSince: "2026-10-02T00:00:00Z", signal });
  const [path, options] = request.mock.calls[0];
  const url = new URL(path, "http://127.0.0.1");
  expect(url.pathname).toBe("/api/recovery-items");
  expect(url.searchParams.get("cursor")).toBe("next+/=");
  expect(url.searchParams.get("state")).toBe("blocked");
  expect(url.searchParams.get("kind")).toBe("media_library_entry");
  expect(url.searchParams.get("deleted_since")).toBe("2026-10-02T00:00:00Z");
  expect(options.signal).toBe(signal);
});

it("keeps media membership identities encoded and sends only its frozen command", async () => {
  const request = vi.fn().mockResolvedValue({});
  const api = recoveryApi(request);
  const signal = new AbortController().signal;
  const command = { expected_revision: "a".repeat(64), impact_sha256: "b".repeat(64), operation_key: "trash-media" };
  await api.mediaDeletionImpact("libentry:sha256:bytes/segment", signal);
  expect(request).toHaveBeenLastCalledWith("/api/artifact-library/libentry%3Asha256%3Abytes%2Fsegment/deletion-impact", { signal });
  await api.trashMedia("libentry:sha256:bytes/segment", command);
  expect(request).toHaveBeenLastCalledWith("/api/artifact-library/libentry%3Asha256%3Abytes%2Fsegment/trash", { method: "POST", body: JSON.stringify(command) });
});

it("binds deletion previews and trash to the same chat and media choice", async () => {
  const request = vi.fn().mockResolvedValue({});
  const api = recoveryApi(request);
  const signal = new AbortController().signal;
  const command = { expected_revision: "a".repeat(64), impact_sha256: "b".repeat(64), operation_key: "trash-garden", delete_generated_media: true };
  await api.deletionImpact("chat/garden", true, signal);
  expect(request).toHaveBeenLastCalledWith("/api/chats/chat%2Fgarden/deletion-impact?delete_generated_media=true", { signal });
  await api.trashChat("chat/garden", command);
  expect(request).toHaveBeenLastCalledWith("/api/chats/chat%2Fgarden/trash", { method: "POST", body: JSON.stringify(command) });
});

it("keeps recovery identifiers in one path segment and retains complete replay commands", async () => {
  const request = vi.fn().mockResolvedValue({});
  const api = recoveryApi(request);
  const command = { expected_revision: "a".repeat(64), impact_sha256: "b".repeat(64), operation_key: "restore-garden" };
  await api.recoveryImpact("deleted/garden");
  expect(request).toHaveBeenLastCalledWith("/api/recovery-items/deleted%2Fgarden/impact", { signal: undefined });
  await api.restoreRecovery("deleted/garden", { ...command, restore_unfiled: true });
  expect(request).toHaveBeenLastCalledWith("/api/recovery-items/deleted%2Fgarden/restore", { method: "POST", body: JSON.stringify({ ...command, restore_unfiled: true }) });
  await api.purgeRecovery("deleted/garden", { ...command, acknowledgement: "permanently-delete" });
  expect(request).toHaveBeenLastCalledWith("/api/recovery-items/deleted%2Fgarden/purge", { method: "POST", body: JSON.stringify({ ...command, acknowledgement: "permanently-delete" }) });
});
