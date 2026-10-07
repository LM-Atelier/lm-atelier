/** Clearing, from Data & backups, the media retention would clear at the next start. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { StorageSummary } from "./StorageSummary";
import type { ArtifactCleanupResult, ArtifactStorageInfo } from "./types";

vi.mock("./api", () => ({
  api: { artifactStorage: vi.fn(), modelStorage: vi.fn(), cleanupArtifacts: vi.fn() },
}));

const MIB = 1024 ** 2;

function media(overrides: Partial<ArtifactStorageInfo> = {}): ArtifactStorageInfo {
  return {
    total_bytes: 900 * MIB,
    total_count: 60,
    referenced_bytes: 600 * MIB,
    referenced_count: 40,
    unreferenced_bytes: 300 * MIB,
    unreferenced_count: 20,
    temporary_bytes: 100 * MIB,
    temporary_count: 10,
    eligible_bytes: 256 * MIB,
    eligible_count: 12,
    retention_pending_count: 0,
    disk_free_bytes: 50 * 1024 * MIB,
    warning: false,
    retention_days: 30,
    temporary_retention_hours: 24,
    ...overrides,
  };
}

function batch(overrides: Partial<ArtifactCleanupResult> = {}): ArtifactCleanupResult {
  return {
    dry_run: false,
    marked_count: 0,
    retention_pending_count: 0,
    removed_count: 0,
    reclaimed_bytes: 0,
    truncated: false,
    ...overrides,
  };
}

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><StorageSummary /></QueryClientProvider>);
}

function realRuns(): number {
  return vi.mocked(api.cleanupArtifacts).mock.calls.filter(([dryRun]) => dryRun === false).length;
}

beforeEach(() => {
  vi.mocked(api.modelStorage).mockResolvedValue({
    installed_bytes: 0,
    partial_download_bytes: 0,
    catalog_cache_bytes: 0,
    installed_count: 0,
    partial_download_count: 0,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("offers nothing to clear when nothing is ready", async () => {
  vi.mocked(api.artifactStorage).mockResolvedValue(media({ eligible_count: 0, eligible_bytes: 0 }));
  show();

  await screen.findByText("Ready to clear", { selector: "dt" });
  expect(screen.queryByRole("button", { name: "Clear now" })).toBeNull();
});

it("asks with a fresh preview and the rule, and clears nothing when cancelled", async () => {
  vi.mocked(api.artifactStorage).mockResolvedValue(media());
  // More is ready now than when the summary was read: the question shows now.
  vi.mocked(api.cleanupArtifacts).mockResolvedValue(
    batch({ dry_run: true, removed_count: 14, reclaimed_bytes: 300 * MIB }),
  );
  show();

  fireEvent.click(await screen.findByRole("button", { name: "Clear now" }));
  const dialog = await screen.findByRole("dialog", { name: "Clear media nothing uses?" });
  expect(dialog).toHaveTextContent("14 files, 300 MB.");
  expect(dialog).toHaveTextContent("once they are 24 hours old");
  expect(dialog).toHaveTextContent("once it has gone 30 days unused");
  expect(api.cleanupArtifacts).toHaveBeenCalledWith(true);

  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(realRuns()).toBe(0);
});

it("clears batch after batch until the server says it is done, then refreshes the figures", async () => {
  vi.mocked(api.artifactStorage).mockResolvedValue(media());
  vi.mocked(api.cleanupArtifacts)
    .mockResolvedValueOnce(batch({ dry_run: true, removed_count: 1005, reclaimed_bytes: 2 * 1024 * MIB }))
    .mockResolvedValueOnce(batch({ removed_count: 1000, reclaimed_bytes: 1024 * MIB, truncated: true }))
    .mockResolvedValueOnce(batch({ removed_count: 5, reclaimed_bytes: 1024 * MIB }));
  show();

  fireEvent.click(await screen.findByRole("button", { name: "Clear now" }));
  const dialog = await screen.findByRole("dialog", { name: "Clear media nothing uses?" });
  const readsBefore = vi.mocked(api.artifactStorage).mock.calls.length;
  fireEvent.click(within(dialog).getByRole("button", { name: "Clear" }));

  expect(await screen.findByRole("status")).toHaveTextContent("Cleared 1,005 files, 2.0 GB.");
  expect(realRuns()).toBe(2);
  expect(screen.queryByRole("dialog")).toBeNull();
  await waitFor(() => expect(vi.mocked(api.artifactStorage).mock.calls.length).toBeGreaterThan(readsBefore));
});

it("stops when a batch reports more to do but removed nothing", async () => {
  vi.mocked(api.artifactStorage).mockResolvedValue(media());
  let stalled = 0;
  vi.mocked(api.cleanupArtifacts).mockImplementation(async (dryRun) => {
    if (dryRun) return batch({ dry_run: true, removed_count: 3, reclaimed_bytes: MIB });
    // Answers "more to do" a few times without progress, then finishes, so a
    // run that keeps asking shows up as extra calls rather than a hang.
    stalled += 1;
    return batch({ removed_count: 0, truncated: stalled < 4 });
  });
  show();

  fireEvent.click(await screen.findByRole("button", { name: "Clear now" }));
  fireEvent.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Clear" }));

  expect(await screen.findByRole("status")).toHaveTextContent("Nothing needed clearing after all.");
  expect(realRuns()).toBe(1);
});

it("says so, without asking, when the preview finds nothing left", async () => {
  vi.mocked(api.artifactStorage).mockResolvedValue(media());
  vi.mocked(api.cleanupArtifacts).mockResolvedValue(batch({ dry_run: true }));
  show();

  fireEvent.click(await screen.findByRole("button", { name: "Clear now" }));

  expect(await screen.findByRole("status")).toHaveTextContent("Nothing is ready to clear any more.");
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(realRuns()).toBe(0);
});

it("reports a failure part way through, and still refreshes what was cleared", async () => {
  vi.mocked(api.artifactStorage).mockResolvedValue(media());
  vi.mocked(api.cleanupArtifacts)
    .mockResolvedValueOnce(batch({ dry_run: true, removed_count: 1005, reclaimed_bytes: 2 * 1024 * MIB }))
    .mockResolvedValueOnce(batch({ removed_count: 1000, reclaimed_bytes: 1024 * MIB, truncated: true }))
    .mockRejectedValueOnce(new Error("The database is busy."));
  show();

  fireEvent.click(await screen.findByRole("button", { name: "Clear now" }));
  const dialog = await screen.findByRole("dialog");
  const readsBefore = vi.mocked(api.artifactStorage).mock.calls.length;
  fireEvent.click(within(dialog).getByRole("button", { name: "Clear" }));

  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Clearing stopped before it finished, and anything already cleared stays cleared.");
  expect(alert).toHaveTextContent("The database is busy.");
  await waitFor(() => expect(vi.mocked(api.artifactStorage).mock.calls.length).toBeGreaterThan(readsBefore));
});
