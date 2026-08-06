import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MediaLibraryView } from "./MediaLibraryView";
import { api } from "./api";

vi.mock("./api", () => ({
  api: { artifacts: vi.fn(), storageUsage: vi.fn().mockResolvedValue(null) },
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
