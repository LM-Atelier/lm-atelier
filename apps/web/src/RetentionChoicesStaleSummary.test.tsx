import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { StorageSummary } from "./StorageSummary";
import type { ArtifactStorageInfo, RetentionPolicy } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      artifactStorage: vi.fn(), modelStorage: vi.fn(), cleanupArtifacts: vi.fn(),
      retentionPolicy: vi.fn(), chooseRetention: vi.fn(), previewRetention: vi.fn(),
    },
  };
});

afterEach(() => { cleanup(); vi.clearAllMocks(); });

it("asks before newly eligible files replace files cleared since the displayed summary", async () => {
  const current: RetentionPolicy = {
    media_days: 30, temporary_hours: 24, revision: 0,
    default_media_days: 30, default_temporary_hours: 24,
  };
  const displayed: ArtifactStorageInfo = {
    total_bytes: 30, total_count: 3, referenced_bytes: 0, referenced_count: 0,
    unreferenced_bytes: 30, unreferenced_count: 3, temporary_bytes: 0, temporary_count: 0,
    eligible_bytes: 20, eligible_count: 2, retention_pending_count: 1,
    disk_free_bytes: 100_000_000, warning: false, retention_days: 30,
    temporary_retention_hours: 24,
  };
  vi.mocked(api.artifactStorage).mockResolvedValueOnce(displayed).mockResolvedValue({
    ...displayed, total_count: 1, eligible_count: 0, eligible_bytes: 0,
  });
  vi.mocked(api.modelStorage).mockResolvedValue({
    installed_bytes: 0, partial_download_bytes: 0, catalog_cache_bytes: 0,
    installed_count: 0, partial_download_count: 0,
  });
  vi.mocked(api.retentionPolicy).mockResolvedValue(current);
  vi.mocked(api.previewRetention).mockResolvedValue({
    dry_run: true, removed_count: 1, reclaimed_bytes: 10,
    marked_count: 0, retention_pending_count: 0, truncated: false,
  });
  vi.mocked(api.chooseRetention).mockResolvedValue({ ...current, revision: 1, media_days: 7 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><StorageSummary /></QueryClientProvider>);

  // Another window cleared the two old files without changing the policy.
  // The server preview now counts the one ten-day-old file newly eligible at seven days.
  fireEvent.change(await screen.findByRole("combobox", { name: "Images and videos" }), {
    target: { value: "7" },
  });
  await waitFor(() => expect(
    screen.queryByRole("dialog") || vi.mocked(api.chooseRetention).mock.calls.length,
  ).toBeTruthy());
  expect(api.chooseRetention).not.toHaveBeenCalled();
  expect(screen.getByRole("dialog")).toBeInTheDocument();
});
