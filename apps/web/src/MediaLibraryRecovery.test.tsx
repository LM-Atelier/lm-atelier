import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { parseArtifactLibraryPage } from "./artifactLibraryPage";
import { MediaLibraryView } from "./MediaLibraryView";
import { recoveryImpact, recoveryItem } from "./test/recoveryFixtures";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: {
    artifactLibrary: vi.fn(), favoriteArtifact: vi.fn(), mediaDeletionImpact: vi.fn(), trashMedia: vi.fn(),
    recoveryImpact: vi.fn(), restoreRecovery: vi.fn(),
  } };
});

const artifactId = `sha256:${"a".repeat(64)}`;
const entryId = `libentry:${artifactId}`;
const entry = { id: entryId, artifact_id: artifactId, version: 1, state: "visible", display_name: "Garden drawing",
  favorite: true, kind: "image", media_type: "image/png", size_bytes: 1024,
  created_at: "2026-10-02T00:00:00Z", updated_at: "2026-10-02T00:00:00Z" };
const item = { ...recoveryItem(entryId), kind: "media_library_entry" as const, display_label: entry.display_name };
const preview = { ...recoveryImpact(entryId), kind: item.kind };
const restorePreview = { ...preview, revision: "d".repeat(64), available_actions: ["restore", "purge"] as ("restore" | "purge")[] };
const clients: QueryClient[] = [];

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><MediaLibraryView /></QueryClientProvider>);
  return client;
}

async function choose() {
  fireEvent.click(await screen.findByRole("button", { name: "Move Garden drawing to Recently Deleted" }));
  const dialog = screen.getByRole("dialog");
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  return dialog;
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.artifactLibrary).mockResolvedValue(parseArtifactLibraryPage({ items: [entry], next_cursor: null }, 20));
  vi.mocked(api.mediaDeletionImpact).mockResolvedValue(preview);
  vi.mocked(api.trashMedia).mockResolvedValue(item);
  vi.mocked(api.recoveryImpact).mockResolvedValue(restorePreview);
  vi.mocked(api.restoreRecovery).mockResolvedValue({ deletion_id: item.deletion_id, kind: item.kind,
    subject_id: entryId, action: "restore", replayed: false, reclaimed_bytes: 0 });
});
afterEach(() => { cleanup(); for (const client of clients.splice(0)) client.clear(); vi.useRealTimers(); });

it("previews only the selected membership and leaves Cancel inert", async () => {
  show();
  const dialog = await choose();
  expect(api.mediaDeletionImpact).toHaveBeenCalledWith(entryId, expect.any(AbortSignal));
  expect(within(dialog).getByText(/No media bytes are removed now/)).toBeTruthy();
  expect(api.trashMedia).not.toHaveBeenCalled();
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.getByText(entry.display_name)).toBeTruthy();
});

it("keeps a pending confirmation and prevents duplicate submission or cancellation", async () => {
  let finish!: (value: typeof item) => void;
  vi.mocked(api.trashMedia).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  show();
  const dialog = await choose();
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  await waitFor(() => expect(api.trashMedia).toHaveBeenCalledTimes(1));
  fireEvent.click(within(dialog).getByRole("button", { name: "Moving item…" }));
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(screen.getByRole("dialog")).toBeTruthy();
  expect(api.trashMedia).toHaveBeenCalledTimes(1);
  vi.mocked(api.artifactLibrary).mockResolvedValue(parseArtifactLibraryPage({ items: [], next_cursor: null }, 20));
  finish(item);
  expect(await screen.findByRole("button", { name: "Undo" })).toBeTruthy();
  await screen.findByText("No media matches these filters");
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("requires a fresh preview and another confirmation after a known refusal", async () => {
  vi.mocked(api.trashMedia).mockRejectedValueOnce(new ApiError(409, undefined, "The item changed.", "recovery-impact-stale"));
  show();
  const dialog = await choose();
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  await screen.findByText("The item changed.");
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  expect(api.trashMedia).toHaveBeenCalledTimes(1);
  vi.mocked(api.mediaDeletionImpact).mockResolvedValue({ ...preview, revision: "e".repeat(64) });
  fireEvent.click(within(dialog).getByRole("button", { name: "Check again" }));
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  expect(api.trashMedia).toHaveBeenCalledTimes(1);
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  await screen.findByRole("button", { name: "Undo" });
  expect(vi.mocked(api.trashMedia).mock.calls[1][1].expected_revision).toBe("e".repeat(64));
});

it("retries an uncertain Trash response with the identical frozen command", async () => {
  vi.mocked(api.trashMedia).mockRejectedValueOnce(new Error("Connection interrupted."));
  show();
  const dialog = await choose();
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  await screen.findByText("Connection interrupted.");
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  await screen.findByRole("button", { name: "Undo" });
  expect(api.trashMedia).toHaveBeenCalledTimes(2);
  expect(vi.mocked(api.trashMedia).mock.calls[1]).toEqual(vi.mocked(api.trashMedia).mock.calls[0]);
  expect(api.mediaDeletionImpact).toHaveBeenCalledTimes(1);
});

it("retries uncertain Undo with its original command and refreshes the same membership", async () => {
  vi.mocked(api.restoreRecovery).mockRejectedValueOnce(new Error("Connection interrupted."));
  const client = show();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const dialog = await choose();
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  fireEvent.click(await screen.findByRole("button", { name: "Undo" }));
  fireEvent.click(await screen.findByRole("button", { name: "Try Undo again" }));
  await waitFor(() => expect(screen.queryByRole("button", { name: "Undo" })).toBeNull());
  expect(api.restoreRecovery).toHaveBeenCalledTimes(2);
  expect(vi.mocked(api.restoreRecovery).mock.calls[1]).toEqual(vi.mocked(api.restoreRecovery).mock.calls[0]);
  expect(api.recoveryImpact).toHaveBeenCalledTimes(1);
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["artifact-library-v1"] });
});

it("rechecks a refused Undo without restoring automatically", async () => {
  vi.mocked(api.restoreRecovery).mockRejectedValueOnce(new ApiError(409, undefined, "The item changed.", "recovery-impact-stale"));
  show();
  const dialog = await choose();
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  fireEvent.click(await screen.findByRole("button", { name: "Undo" }));
  await screen.findByText("The item changed.");
  expect(screen.getByRole("button", { name: "Try Undo again" })).toHaveAttribute("aria-disabled", "true");
  vi.mocked(api.recoveryImpact).mockResolvedValue({ ...restorePreview, revision: "f".repeat(64) });
  fireEvent.click(screen.getByRole("button", { name: "Recheck Undo" }));
  expect(api.restoreRecovery).toHaveBeenCalledTimes(1);
  fireEvent.click(await screen.findByRole("button", { name: "Undo" }));
  await waitFor(() => expect(api.restoreRecovery).toHaveBeenCalledTimes(2));
  expect(vi.mocked(api.restoreRecovery).mock.calls[1][1].expected_revision).toBe("f".repeat(64));
});

it("keeps Undo inert when the returned recovery deadline has passed", async () => {
  vi.mocked(api.trashMedia).mockResolvedValue({ ...item, purge_after: "2000-01-01T00:00:00Z" });
  show();
  const dialog = await choose();
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  const undo = await screen.findByRole("button", { name: "Undo" });
  expect(undo).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(undo);
  expect(api.recoveryImpact).not.toHaveBeenCalled();
  expect(api.restoreRecovery).not.toHaveBeenCalled();
});
