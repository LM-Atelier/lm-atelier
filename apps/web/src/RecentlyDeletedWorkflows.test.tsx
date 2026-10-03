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
const item: RecoveryItem = { ...recoveryItem("workflow-garden"), kind: "workflow_family", display_label: "Garden layout",
  counts: { ...recoveryItem().counts, workflow_families: 1, workflow_definitions: 2, workflow_revisions: 3, references: 4 } };
const preview: RecoveryImpact = { ...recoveryImpact(item.subject_id, ["restore"]), kind: "workflow_family", counts: item.counts };
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
  vi.mocked(api.restoreRecovery).mockResolvedValue({ kind: "workflow_family", subject_id: item.subject_id,
    deletion_id: item.deletion_id, action: "restore", replayed: false, reclaimed_bytes: 0 });
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

it("filters workflows on the server and shows revision and reference impact", async () => {
  show();
  await screen.findByText("Garden layout");
  fireEvent.change(screen.getByLabelText("Type"), { target: { value: "workflow_family" } });
  await waitFor(() => expect(api.recoveryItems).toHaveBeenLastCalledWith(expect.objectContaining({ kind: "workflow_family" })));
  const row = (await screen.findByText("Garden layout")).closest("article")!;
  expect(within(row).getByText(/2 workflows · 3 revisions/)).toBeVisible();
  expect(within(row).getByText(/4 retained references/)).toBeVisible();
  expect(within(row).getByText(/Workflow/)).toBeVisible();
  expect(screen.queryByText(/0 messages · 0 media files/)).toBeNull();
});

it("restores without enabling or activating a workflow and refreshes its selectors", async () => {
  const client = show();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  fireEvent.click(await screen.findByRole("button", { name: "Restore Garden layout" }));
  const dialog = await screen.findByRole("dialog", { name: "Restore this workflow?" });
  expect(within(dialog).getByText(/without enabling it, granting trust or activating it/)).toBeVisible();
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Restore workflow" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore workflow" }));
  await screen.findByText("Workflow restored. It remains disabled until you review and enable it.");
  expect(api.restoreRecovery).toHaveBeenCalledWith(item.deletion_id, expect.objectContaining({ expected_revision: preview.revision }));
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["workflow-families"] });
});

it("refuses permanent deletion when the current impact requires historical revisions", async () => {
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Permanently delete Garden layout" }));
  const dialog = await screen.findByRole("dialog", { name: "Permanently delete this workflow?" });
  expect(within(dialog).getByText(/Shared models and media remain available/)).toBeVisible();
  await within(dialog).findByText(/This action is unavailable/);
  expect(within(dialog).getByRole("button", { name: "Delete permanently" })).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(within(dialog).getByRole("button", { name: "Delete permanently" }));
  expect(api.purgeRecovery).not.toHaveBeenCalled();
});
