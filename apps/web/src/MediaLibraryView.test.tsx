import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MediaLibraryView } from "./MediaLibraryView";
import { api } from "./api";

vi.mock("./api", () => ({
  api: { artifacts: vi.fn(), artifactStorage: vi.fn().mockResolvedValue(null) },
}));

function renderLibrary() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MediaLibraryView />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("a library that cannot be read", () => {
  it("says so rather than reporting the library as empty", async () => {
    // Undefined data and no data read identically at the render, so a failed
    // request told the user their own library held nothing.
    vi.mocked(api.artifacts).mockRejectedValue(new Error("media library unreachable"));
    renderLibrary();

    await waitFor(() => expect(screen.getByText("media library unreachable")).toBeTruthy());
    expect(screen.queryByText("No generated media")).toBeNull();
  });

  it("still reports a genuinely empty library as empty", async () => {
    vi.mocked(api.artifacts).mockResolvedValue([]);
    renderLibrary();

    await waitFor(() => expect(screen.getByText("No generated media")).toBeTruthy());
  });
});

describe("the storage summary", () => {
  it("renders what the library is using and offers cleanup", async () => {
    // This section was never exercised: the mock named a function the
    // component does not call, so the query failed, `storage.data` stayed
    // undefined, and the whole section was absent from every assertion in
    // this file while the file claimed to cover the library.
    vi.mocked(api.artifacts).mockResolvedValue([]);
    vi.mocked(api.artifactStorage).mockResolvedValue({
      total_bytes: 2048,
      total_count: 2,
      referenced_bytes: 1024,
      referenced_count: 1,
      disk_free_bytes: 4096,
      eligible_bytes: 1024,
      eligible_count: 1,
      warning: false,
    } as never);

    renderLibrary();

    await waitFor(() => expect(screen.getByText("2 stored artifacts")).toBeTruthy());
    expect(screen.getByText("1 eligible for cleanup")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Run cleanup" })).toBeEnabled();
  });
});
