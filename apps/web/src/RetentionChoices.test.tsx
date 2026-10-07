/** Choosing, from Data & backups, how long files nothing uses are kept. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { StorageSummary } from "./StorageSummary";
import type { ArtifactCleanupResult, ArtifactStorageInfo, RetentionPolicy } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      artifactStorage: vi.fn(),
      modelStorage: vi.fn(),
      cleanupArtifacts: vi.fn(),
      retentionPolicy: vi.fn(),
      chooseRetention: vi.fn(),
      previewRetention: vi.fn(),
    },
  };
});

const MIB = 1024 ** 2;

function media(): ArtifactStorageInfo {
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
    retention_pending_count: 8,
    disk_free_bytes: 50 * 1024 * MIB,
    warning: false,
    retention_days: 30,
    temporary_retention_hours: 24,
  };
}

function policy(overrides: Partial<RetentionPolicy> = {}): RetentionPolicy {
  return {
    media_days: 30,
    temporary_hours: 24,
    revision: 0,
    default_media_days: 30,
    default_temporary_hours: 24,
    ...overrides,
  };
}

function found(removed: number): ArtifactCleanupResult {
  return {
    dry_run: true,
    marked_count: 0,
    retention_pending_count: 0,
    removed_count: removed,
    reclaimed_bytes: removed * 10 * MIB,
    truncated: false,
  };
}

let client: QueryClient;

function show() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><StorageSummary /></QueryClientProvider>);
}

const mediaChoice = () => screen.findByRole("combobox", { name: "Images and videos" });
const previewChoice = () => screen.findByRole("combobox", { name: "Previews and in-between steps" });

beforeEach(() => {
  vi.mocked(api.artifactStorage).mockResolvedValue(media());
  vi.mocked(api.modelStorage).mockResolvedValue({
    installed_bytes: 0,
    partial_download_bytes: 0,
    catalog_cache_bytes: 0,
    installed_count: 0,
    partial_download_count: 0,
  });
  vi.mocked(api.retentionPolicy).mockResolvedValue(policy());
  vi.mocked(api.chooseRetention).mockImplementation(async (revision, days, hours) =>
    policy({ media_days: days, temporary_hours: hours, revision: revision + 1 }));
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("shows the windows in force, marking the installation's usual ones", async () => {
  vi.mocked(api.retentionPolicy).mockResolvedValue(policy({ media_days: 45, revision: 3 }));
  show();

  const images = await mediaChoice();
  expect(images).toHaveValue("45");
  expect(within(images).getAllByRole("option").map((option) => option.textContent)).toEqual([
    "7 days", "14 days", "30 days (usual)", "45 days", "90 days", "180 days", "1 year",
  ]);
  const previews = await previewChoice();
  expect(previews).toHaveValue("24");
  expect(within(previews).getByRole("option", { name: "1 day (usual)" })).toBeInTheDocument();
  expect(within(previews).getByRole("option", { name: "6 hours" })).toBeInTheDocument();
});

it("keeps files longer at once, without asking or checking", async () => {
  show();

  fireEvent.change(await mediaChoice(), { target: { value: "90" } });

  await waitFor(() => expect(api.chooseRetention).toHaveBeenCalledWith(0, 90, 24));
  expect(api.previewRetention).not.toHaveBeenCalled();
  expect(screen.queryByRole("dialog")).toBeNull();
  await waitFor(async () => expect(await mediaChoice()).toHaveValue("90"));
});

it("saves a shorter window without asking when it would clear nothing", async () => {
  vi.mocked(api.previewRetention).mockResolvedValue(found(0));
  show();

  fireEvent.change(await previewChoice(), { target: { value: "6" } });

  await waitFor(() => expect(api.chooseRetention).toHaveBeenCalledWith(0, 30, 6));
  expect(api.previewRetention).toHaveBeenCalledWith(30, 6);
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("asks before a shorter window that would clear anything, even fewer files than the figures show", async () => {
  // The summary on screen says 12 are ready, but it may be out of date: the
  // server's fresh count is what decides.
  vi.mocked(api.previewRetention).mockResolvedValue(found(3));
  show();

  fireEvent.change(await mediaChoice(), { target: { value: "7" } });

  const dialog = await screen.findByRole("dialog");
  expect(dialog).toHaveTextContent("3 files, 30 MB, would be ready to clear.");
  expect(dialog).not.toHaveTextContent("ready now");
  expect(dialog).toHaveTextContent("Images and videos would be kept 7 days after nothing uses them, and previews 1 day.");
  fireEvent.click(within(dialog).getByRole("button", { name: "Close without changing anything" }));

  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(api.chooseRetention).not.toHaveBeenCalled();
  expect(await mediaChoice()).toHaveValue("30");
});

it("saves the shorter window once confirmed, and refreshes what is ready", async () => {
  vi.mocked(api.previewRetention).mockResolvedValue(found(20));
  show();
  await mediaChoice();
  const storageReads = vi.mocked(api.artifactStorage).mock.calls.length;

  fireEvent.change(await mediaChoice(), { target: { value: "7" } });
  fireEvent.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Keep them for less" }));

  await waitFor(() => expect(api.chooseRetention).toHaveBeenCalledWith(0, 7, 24));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(await mediaChoice()).toHaveValue("7");
  await waitFor(() => expect(vi.mocked(api.artifactStorage).mock.calls.length).toBeGreaterThan(storageReads));
});

it("shows the choice in force when another window changed it first", async () => {
  vi.mocked(api.retentionPolicy)
    .mockResolvedValueOnce(policy())
    .mockResolvedValue(policy({ media_days: 14, temporary_hours: 12, revision: 2 }));
  vi.mocked(api.chooseRetention).mockRejectedValue(
    new ApiError(409, "stale", "stale", "retention-policy-stale"),
  );
  show();

  fireEvent.change(await mediaChoice(), { target: { value: "90" } });

  expect(await screen.findByRole("alert")).toHaveTextContent("Retention was changed in another window, so nothing was saved.");
  await waitFor(async () => expect(await mediaChoice()).toHaveValue("14"));
  expect(await previewChoice()).toHaveValue("12");
});

it("names the revision the choice was made from, even after the choice in force is read again", async () => {
  let answer: (result: ArtifactCleanupResult) => void = () => undefined;
  vi.mocked(api.previewRetention).mockReturnValue(new Promise((resolve) => { answer = resolve; }));
  vi.mocked(api.retentionPolicy)
    .mockResolvedValueOnce(policy())
    .mockResolvedValue(policy({ revision: 5 }));
  show();

  fireEvent.change(await mediaChoice(), { target: { value: "7" } });
  await act(async () => {
    await client.refetchQueries({ queryKey: ["retention-policy"] });
  });
  expect(client.getQueryData<RetentionPolicy>(["retention-policy"])?.revision).toBe(5);
  await act(async () => answer(found(0)));

  await waitFor(() => expect(api.chooseRetention).toHaveBeenCalledTimes(1));
  expect(api.chooseRetention).toHaveBeenCalledWith(0, 7, 24);
});

it("lets one choice finish before taking another", async () => {
  let finish: (result: RetentionPolicy) => void = () => undefined;
  vi.mocked(api.chooseRetention).mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
  show();

  fireEvent.change(await mediaChoice(), { target: { value: "90" } });
  await waitFor(() => expect(api.chooseRetention).toHaveBeenCalledTimes(1));
  fireEvent.change(await previewChoice(), { target: { value: "72" } });
  expect(api.chooseRetention).toHaveBeenCalledTimes(1);
  expect(await previewChoice()).toHaveAttribute("aria-disabled", "true");

  await act(async () => finish(policy({ media_days: 90, revision: 1 })));
  await waitFor(async () => expect(await previewChoice()).toHaveAttribute("aria-disabled", "false"));
});

it("keeps the storage figures when the choices cannot be read", async () => {
  vi.mocked(api.retentionPolicy).mockRejectedValue(new Error("unreachable"));
  show();

  expect(await screen.findByText("Retention choices are unavailable right now.")).toBeInTheDocument();
  expect(screen.getByText("Ready to clear", { selector: "dt" })).toBeInTheDocument();
  expect(screen.queryByRole("combobox")).toBeNull();
});
