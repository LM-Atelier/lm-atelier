/** Where the workspace's disk space went, as Data & backups shows it. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { StorageSummary } from "./StorageSummary";
import type { ArtifactStorageInfo, ModelStorageInfo } from "./types";

vi.mock("./api", () => ({ api: { artifactStorage: vi.fn(), modelStorage: vi.fn() } }));

const GIB = 1024 ** 3;

function media(overrides: Partial<ArtifactStorageInfo> = {}): ArtifactStorageInfo {
  return {
    total_bytes: 3 * GIB,
    total_count: 120,
    referenced_bytes: 2 * GIB,
    referenced_count: 80,
    unreferenced_bytes: GIB,
    unreferenced_count: 40,
    temporary_bytes: 512 * 1024 ** 2,
    temporary_count: 30,
    eligible_bytes: 256 * 1024 ** 2,
    eligible_count: 12,
    retention_pending_count: 5,
    disk_free_bytes: 200 * GIB,
    warning: false,
    retention_days: 30,
    temporary_retention_hours: 24,
    ...overrides,
  };
}

function models(overrides: Partial<ModelStorageInfo> = {}): ModelStorageInfo {
  return {
    installed_bytes: 40 * GIB,
    partial_download_bytes: 0,
    catalog_cache_bytes: 12 * 1024 ** 2,
    installed_count: 4,
    partial_download_count: 0,
    ...overrides,
  };
}

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><StorageSummary /></QueryClientProvider>);
}

function figure(term: string): string {
  const label = screen.getByText(term, { selector: "dt" });
  return within(label.parentElement as HTMLElement).getByRole("definition").textContent ?? "";
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("shows what models and media use, what is in use, and what could be cleared", async () => {
  vi.mocked(api.artifactStorage).mockResolvedValue(media());
  vi.mocked(api.modelStorage).mockResolvedValue(models());
  show();

  await screen.findByText("Images and videos", { selector: "dt" });
  expect(figure("Models")).toBe("4 installed models · 40 GB");
  expect(figure("Images and videos")).toBe("120 files · 3.0 GB");
  expect(figure("In use")).toMatch(/^80 files · 2\.0 GB/);
  expect(figure("Previews and in-between steps")).toMatch(/24 hours old/);
  expect(figure("Ready to clear")).toMatch(/^12 files · 256 MB.*30 days unused/);
  expect(figure("Waiting to be cleared")).toMatch(/^5 files.*less than 30 days/);
  expect(figure("Free disk space")).toBe("200 GB");
  // Nothing to report about unfinished downloads, so nothing is shown for them.
  expect(screen.queryByText("Unfinished downloads")).toBeNull();
  expect(screen.queryByRole("alert")).toBeNull();
});

it("lists unfinished downloads only when there are some, and warns when the disk is nearly full", async () => {
  vi.mocked(api.artifactStorage).mockResolvedValue(media({ warning: true, disk_free_bytes: 2 * GIB }));
  vi.mocked(api.modelStorage).mockResolvedValue(
    models({ partial_download_count: 1, partial_download_bytes: 3 * GIB }),
  );
  show();

  expect(await screen.findByRole("alert")).toHaveTextContent("Free disk space is low: 2.0 GB left.");
  expect(figure("Unfinished downloads")).toBe("1 download · 3.0 GB");
});

it("says the figures are unavailable rather than showing an empty page", async () => {
  vi.mocked(api.artifactStorage).mockRejectedValue(new Error("unavailable"));
  vi.mocked(api.modelStorage).mockResolvedValue(models());
  show();

  expect(await screen.findByRole("alert")).toHaveTextContent("Storage figures are unavailable right now.");
});
