import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MediaLibraryView } from "./MediaLibraryView";
import { api } from "./api";
import { parseArtifactLibraryPage, type ArtifactLibraryFilters } from "./artifactLibraryPage";

vi.mock("./api", () => ({
  api: { artifactLibrary: vi.fn(), favoriteArtifact: vi.fn() },
}));

const digest = (character: string) => /^[a-f]$/.test(character)
  ? character.repeat(64)
  : character.charCodeAt(0).toString(16).padStart(64, "0");
const cursor = `cGF5bG9hZA.${"a".repeat(43)}`;

function rawItem(character: string, options: {
  createdAt?: string;
  favorite?: boolean;
  kind?: "image" | "video";
} = {}) {
  const sha = digest(character);
  const kind = options.kind ?? "image";
  const createdAt = options.createdAt ?? "2026-08-12T12:00:00Z";
  return {
    id: `libentry:sha256:${sha}`,
    artifact_id: `sha256:${sha}`,
    version: 1,
    state: "visible",
    display_name: `Item ${character}`,
    favorite: options.favorite ?? false,
    kind,
    media_type: `${kind}/png`,
    size_bytes: 1024,
    created_at: createdAt,
    updated_at: createdAt,
  };
}

function parsedPage(items: ReturnType<typeof rawItem>[], nextCursor: string | null = null, limit = 20) {
  return parseArtifactLibraryPage({ items, next_cursor: nextCursor }, limit);
}

function renderLibrary(onEditImage?: (artifactId: string) => void) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MediaLibraryView onEditImage={onEditImage} />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("EntryV1 Media Library feed", () => {
  it("reports a read failure generically and never calls legacy list/delete/cleanup APIs", async () => {
    vi.mocked(api.artifactLibrary).mockRejectedValue(new Error("private backend marker"));
    renderLibrary();

    expect(await screen.findByText("The Media Library could not be loaded safely. Refresh and try again.")).toBeVisible();
    expect(screen.queryByText("private backend marker")).toBeNull();
    expect(screen.queryByRole("button", { name: /delete/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /cleanup/i })).toBeNull();
  });

  it("renders a valid empty page only after the request succeeds", async () => {
    vi.mocked(api.artifactLibrary).mockResolvedValue(parsedPage([]));
    renderLibrary();

    expect(screen.getByLabelText("Loading your media")).toBeVisible();
    expect(await screen.findByText("No media matches these filters")).toBeVisible();
  });

  it("bounds search by backend Unicode code points instead of UTF-16 units", async () => {
    vi.mocked(api.artifactLibrary).mockResolvedValue(parsedPage([]));
    renderLibrary();
    await screen.findByText("No media matches these filters");

    const input = screen.getByRole("textbox", { name: "Search media" });
    fireEvent.change(input, { target: { value: "😀".repeat(201) } });
    expect(input).toHaveValue("😀".repeat(200));
  });

  it("loads a cursor page, states truncation, and preserves exact cursor order", async () => {
    const first = parsedPage(
      Array.from({ length: 20 }, (_, index) => rawItem(String.fromCharCode(122 - index))),
      cursor,
    );
    const second = parsedPage([rawItem("a", { createdAt: "2026-08-12T11:59:00Z", kind: "video" })]);
    vi.mocked(api.artifactLibrary)
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce(second);
    renderLibrary();

    expect(await screen.findByText("Showing the newest 20 items. More are available.")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    await waitFor(() => expect(api.artifactLibrary).toHaveBeenLastCalledWith(
      { kind: "", query: "", favorite: false },
      cursor,
      20,
      expect.any(AbortSignal),
    ));
    expect(await screen.findByLabelText("Item a")).toBeVisible();
    expect(screen.queryByText(/More are available/)).toBeNull();
  });

  it("fails the whole chain on a duplicate page instead of partially appending", async () => {
    const firstItems = Array.from({ length: 20 }, (_, index) => rawItem(String.fromCharCode(122 - index)));
    vi.mocked(api.artifactLibrary)
      .mockResolvedValueOnce(parsedPage(firstItems, cursor))
      .mockResolvedValueOnce(parsedPage([firstItems[19]]));
    renderLibrary();

    await screen.findByText("Item z");
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("The Media Library could not be loaded safely. Refresh and try again.")).toBeVisible();
    expect(screen.queryByText("Item z")).toBeNull();
  });

  it("hands one validated image Artifact ID to Studio without fabricating a legacy DTO", async () => {
    vi.mocked(api.artifactLibrary).mockResolvedValue(parsedPage([rawItem("a")]));
    const onEditImage = vi.fn();
    renderLibrary(onEditImage);

    fireEvent.click(await screen.findByRole("button", { name: "Edit Item a" }));
    expect(onEditImage).toHaveBeenCalledWith(`sha256:${digest("a")}`);
    expect(onEditImage).toHaveBeenCalledTimes(1);
  });

  it("restarts after favorite settlement and never changes the card optimistically", async () => {
    vi.mocked(api.artifactLibrary)
      .mockResolvedValueOnce(parsedPage([rawItem("a")]))
      .mockResolvedValueOnce(parsedPage([rawItem("a", { favorite: true })]));
    vi.mocked(api.favoriteArtifact).mockResolvedValue({} as never);
    renderLibrary();

    fireEvent.click(await screen.findByRole("button", { name: "Favorite Item a" }));
    expect(screen.getByRole("button", { name: "Favorite Item a" })).toHaveAttribute("aria-pressed", "false");
    await waitFor(() => expect(api.artifactLibrary).toHaveBeenCalledTimes(2));
    expect(await screen.findByRole("button", { name: "Unfavorite Item a" })).toBeVisible();
  });

  it("hides prior rows when refresh fails and suppresses private failure text", async () => {
    vi.mocked(api.artifactLibrary)
      .mockResolvedValueOnce(parsedPage([rawItem("a")]))
      .mockRejectedValueOnce(new Error("private refresh marker"));
    renderLibrary();

    await screen.findByText("Item a");
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(await screen.findByText("The Media Library could not be loaded safely. Refresh and try again.")).toBeVisible();
    expect(screen.queryByText("Item a")).toBeNull();
    expect(screen.queryByText("private refresh marker")).toBeNull();
  });

  it("keeps delayed A/B/A results out of the current generation", async () => {
    const requests: Array<{
      filters: ArtifactLibraryFilters;
      signal: AbortSignal;
      resolve: (value: ReturnType<typeof parsedPage>) => void;
    }> = [];
    vi.mocked(api.artifactLibrary).mockImplementation((filters, _cursor, _limit, signal) => new Promise((resolve) => {
      requests.push({ filters, signal: signal!, resolve });
    }));
    renderLibrary();
    await waitFor(() => expect(requests).toHaveLength(1));

    fireEvent.change(screen.getByRole("combobox", { name: "Media type" }), { target: { value: "image" } });
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(requests[0].signal.aborted).toBe(true);
    fireEvent.change(screen.getByRole("combobox", { name: "Media type" }), { target: { value: "" } });
    await waitFor(() => expect(requests).toHaveLength(3));
    expect(requests[1].signal.aborted).toBe(true);

    requests[0].resolve(parsedPage([rawItem("a")]));
    requests[1].resolve(parsedPage([rawItem("b")]));
    requests[2].resolve(parsedPage([rawItem("c")]));
    expect(await screen.findByText("Item c")).toBeVisible();
    expect(screen.queryByText("Item a")).toBeNull();
    expect(screen.queryByText("Item b")).toBeNull();
  });
});
