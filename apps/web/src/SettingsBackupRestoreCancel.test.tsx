/** Cancelling a backup restore scheduled from the recovery backup list. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { SettingsView } from "./SettingsView";
import { useAppearance } from "./theme";
import type { BackupInfo } from "./types";
import { useAppNavigation } from "./useAppNavigation";

vi.mock("./api", async (original) => ({ ApiError: (await original<typeof import("./api")>()).ApiError, api: {
  recoveryItems: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
  system: vi.fn().mockResolvedValue(null),
  about: vi.fn().mockResolvedValue(null),
  profilesPage: vi.fn().mockResolvedValue([]),
  presetsPage: vi.fn().mockResolvedValue([]),
  workers: vi.fn().mockResolvedValue([]),
  runtimes: vi.fn().mockResolvedValue([]),
  backups: vi.fn(),
  backupRestoreState: vi.fn().mockResolvedValue({ state: "none", backup: null, reason: null, failed_at: null, encrypted: false }),
  cancelRestore: vi.fn(),
  credentialStatus: vi.fn().mockResolvedValue({ configured: false, vault_available: true }),
  workerSettings: vi.fn().mockResolvedValue({ worker_startup_seconds: 60 }),
  artifactStorage: vi.fn().mockResolvedValue(null),
  modelStorage: vi.fn().mockResolvedValue(null),
  projects: vi.fn().mockResolvedValue([]),
} }));

const BACKUP: BackupInfo = {
  name: "local-lm-20260110T120000Z-00000001.sqlite3",
  size_bytes: 2048,
  sha256: "abcdef0123456789",
  created_at: "2026-01-10T12:00:00Z",
  verified: true,
  restore_pending: true,
  media_included: false,
  media_size_bytes: 0,
};
const SCHEDULED = "Restore scheduled. Restart LM Atelier to apply the selected backup.";
const clients: QueryClient[] = [];

function NavigationSettings() {
  const navigation = useAppNavigation();
  const appearance = useAppearance();
  return <SettingsView engines={[]} appearance={appearance} destinationId={navigation.settingsDestination}
    onDestinationChange={navigation.setSettingsDestination} focusRequest={navigation.settingsFocusRequest} />;
}

beforeEach(() => {
  window.history.replaceState(null, "", "/?view=settings");
  sessionStorage.clear();
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  vi.mocked(api.backups).mockReset();
  vi.mocked(api.cancelRestore).mockReset();
});

async function showBackups() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><NavigationSettings /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Data & backups" }));
  expect(await screen.findByText(SCHEDULED)).toBeInTheDocument();
}

it("cancels a scheduled restore, says so and offers the backup again", async () => {
  vi.mocked(api.backups)
    .mockResolvedValueOnce([BACKUP])
    .mockResolvedValue([{ ...BACKUP, restore_pending: false }]);
  vi.mocked(api.cancelRestore).mockResolvedValue(undefined);
  await showBackups();
  expect(screen.getByRole("button", { name: `Restore backup ${BACKUP.name} on restart` })).toBeDisabled();

  fireEvent.click(screen.getByRole("button", { name: "Cancel restore" }));

  expect(await screen.findByText("Restore cancelled. The current data stays.")).toBeInTheDocument();
  expect(api.cancelRestore).toHaveBeenCalledTimes(1);
  await waitFor(() => expect(screen.queryByText(SCHEDULED)).toBeNull());
  expect(screen.getByRole("button", { name: `Restore backup ${BACKUP.name} on restart` })).toBeEnabled();
});

it("keeps the scheduled restore when it cannot be cancelled", async () => {
  vi.mocked(api.backups).mockResolvedValue([BACKUP]);
  vi.mocked(api.cancelRestore).mockRejectedValue(new Error("refused"));
  await showBackups();

  fireEvent.click(screen.getByRole("button", { name: "Cancel restore" }));

  expect(await screen.findByText("The restore could not be cancelled. Try again.")).toBeInTheDocument();
  expect(screen.getByText(SCHEDULED)).toBeInTheDocument();
  expect(screen.queryByText("Restore cancelled. The current data stays.")).toBeNull();
  expect(screen.getByRole("button", { name: `Restore backup ${BACKUP.name} on restart` })).toBeDisabled();
});
