import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { RecentlyDeleted } from "./RecentlyDeleted";
import { recoveryImpact, recoveryItem } from "./test/recoveryFixtures";
import type { RecoveryImpact, RecoveryItem } from "./recoveryTypes";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: {
    recoveryItems: vi.fn(), recoveryImpact: vi.fn(), restoreRecovery: vi.fn(), purgeRecovery: vi.fn(),
  } };
});
const item: RecoveryItem = { ...recoveryItem("project-garden"), kind: "project", display_label: "Garden layout",
  counts: { ...recoveryItem().counts, chats: 3 } };
const preview: RecoveryImpact = { ...recoveryImpact(item.subject_id, ["restore", "purge"]), kind: "project", counts: item.counts };
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
  vi.mocked(api.restoreRecovery).mockResolvedValue({ kind: "project", subject_id: item.subject_id,
    deletion_id: item.deletion_id, action: "restore", replayed: false, reclaimed_bytes: 0 });
  vi.mocked(api.purgeRecovery).mockResolvedValue({ kind: "project", subject_id: item.subject_id,
    deletion_id: item.deletion_id, action: "purge", replayed: false, reclaimed_bytes: 0 });
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

it("filters deleted projects on the server and shows retained chat impact", async () => {
  show();
  await screen.findByText("Garden layout");
  fireEvent.change(screen.getByLabelText("Type"), { target: { value: "project" } });
  await waitFor(() => expect(api.recoveryItems).toHaveBeenLastCalledWith(expect.objectContaining({ kind: "project" })));
  await screen.findByText("Garden layout");
  const row = screen.getByText("Garden layout").closest("article")!;
  expect(within(row).getByText(/3 chats stay available/)).toBeVisible();
  expect(within(row).getByText(/Project/)).toBeVisible();
  expect(screen.queryByText(/0 messages · 0 media files/)).toBeNull();
});

it("restores project settings without promising to move chats back", async () => {
  const client = show();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  fireEvent.click(await screen.findByRole("button", { name: "Restore Garden layout" }));
  const dialog = await screen.findByRole("dialog", { name: "Restore this project?" });
  expect(within(dialog).getByText(/Chats moved elsewhere stay there/)).toBeVisible();
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Restore project" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore project" }));
  await screen.findByText("Project restored with its original settings. Chats moved elsewhere stay there.");
  expect(api.restoreRecovery).toHaveBeenCalledWith(item.deletion_id, expect.objectContaining({ expected_revision: preview.revision }));
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["projects"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["chat"] });
});

it("confirms Project purge while keeping its chats and media available", async () => {
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Permanently delete Garden layout" }));
  const dialog = await screen.findByRole("dialog", { name: "Permanently delete this project?" });
  expect(within(dialog).getByText(/Its chats and media remain available/)).toBeVisible();
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Delete permanently" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(api.purgeRecovery).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Permanently delete Garden layout" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Delete permanently" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));
  await screen.findByText("Project permanently deleted. Its chats and media remain available.");
  expect(api.purgeRecovery).toHaveBeenCalledWith(item.deletion_id, expect.objectContaining({ acknowledgement: "permanently-delete" }));
});
