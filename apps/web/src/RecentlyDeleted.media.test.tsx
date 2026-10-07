import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { RecentlyDeleted } from "./RecentlyDeleted";
import { recoveryItem, recoveryImpact } from "./test/recoveryFixtures";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: {
    recoveryItems: vi.fn(), recoveryImpact: vi.fn(), restoreRecovery: vi.fn(), purgeRecovery: vi.fn(),
  } };
});

const item = { ...recoveryItem(), kind: "media_library_entry" as const,
  deletion_id: "deleted-garden-media", subject_id: "libentry:garden", display_label: "Garden drawing",
  original_location: { project_id: null, project_label: null },
};
const preview = { ...recoveryImpact(item.subject_id, ["restore", "purge"]), kind: item.kind };
const clients: QueryClient[] = [];

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><RecentlyDeleted /></QueryClientProvider>);
  return client;
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.recoveryItems).mockResolvedValue({ items: [item], next_cursor: null });
  vi.mocked(api.recoveryImpact).mockResolvedValue(preview);
  vi.mocked(api.restoreRecovery).mockResolvedValue({ deletion_id: item.deletion_id, kind: item.kind,
    subject_id: item.subject_id, action: "restore", replayed: false, reclaimed_bytes: 0 });
  vi.mocked(api.purgeRecovery).mockResolvedValue({ deletion_id: item.deletion_id, kind: item.kind,
    subject_id: item.subject_id, action: "purge", replayed: false, reclaimed_bytes: 0 });
});
afterEach(() => { cleanup(); for (const client of clients.splice(0)) client.clear(); });

it("filters by type before requesting a page and describes library restoration", async () => {
  const client = show();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const row = (await screen.findByText("Garden drawing")).closest("article")!;
  expect(within(row).getByText("Recoverable · Media Library")).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Type"), { target: { value: "media_library_entry" } });
  await waitFor(() => expect(api.recoveryItems).toHaveBeenLastCalledWith(expect.objectContaining({ kind: item.kind, cursor: undefined })));
  fireEvent.click(await screen.findByRole("button", { name: "Restore Garden drawing" }));
  expect(screen.getByRole("heading", { name: "Restore this Media Library item?" })).toBeTruthy();
  expect(screen.getByText(/with its favorites, collections and tags/)).toBeTruthy();
  expect(screen.queryByText(/original history/)).toBeNull();
  await waitFor(() => expect(screen.getByRole("button", { name: "Restore item" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(screen.getByRole("button", { name: "Restore item" }));
  expect(await screen.findByText("Media Library item restored with its favorites, collections and tags.")).toBeTruthy();
  expect(api.restoreRecovery).toHaveBeenCalledWith(item.deletion_id, {
    expected_revision: preview.revision, impact_sha256: preview.impact_sha256,
    operation_key: expect.any(String), restore_unfiled: false,
  });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["artifact-library-v1"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["artifacts"] });
});

it("confirms membership removal without promising immediate file deletion", async () => {
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Permanently delete Garden drawing" }));
  expect(screen.getByRole("heading", { name: "Permanently remove this Media Library item?" })).toBeTruthy();
  expect(screen.getByText(/from the Media Library\? This cannot be undone/)).toBeTruthy();
  expect(screen.getByText(/No immediate storage reclamation is promised/)).toBeTruthy();
  expect(api.purgeRecovery).not.toHaveBeenCalled();
  await waitFor(() => expect(screen.getByRole("button", { name: "Delete permanently" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));
  expect(await screen.findByText("Media Library item permanently removed. Shared and retained media remains available.")).toBeTruthy();
  expect(api.purgeRecovery).toHaveBeenCalledWith(item.deletion_id, {
    expected_revision: preview.revision, impact_sha256: preview.impact_sha256,
    operation_key: expect.any(String), acknowledgement: "permanently-delete",
  });
});
