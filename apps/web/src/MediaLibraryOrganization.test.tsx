import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MediaLibraryView } from "./MediaLibraryView";
import { parseArtifactLibraryPage } from "./artifactLibraryPage";
import { ApiError } from "./api";

const methods = vi.hoisted(() => ({
  artifactLibrary: vi.fn(), favoriteArtifact: vi.fn(),
  mediaOrganizationCatalog: vi.fn(), createMediaCollection: vi.fn(), createMediaTag: vi.fn(),
  previewMediaOrganization: vi.fn(), applyMediaOrganization: vi.fn(),
}));
vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: methods };
});
vi.mock("./ArtifactGenerationDetails", () => ({ ArtifactGenerationDetails: () => null }));
vi.mock("./PictureFileSettings", () => ({ PictureFileSettings: () => null }));
vi.mock("./MediaLibraryRecovery", () => ({ MediaLibraryRecovery: () => null }));

const album = { id: `collection_${"a".repeat(32)}`, kind: "manual", name: "Studies", description: "", version: 1 };
const tag = { id: `mediatag_${"b".repeat(32)}`, slug: "landscape", label: "Landscape", color: null, version: 1 };
const cursor = `cGF5bG9hZA.${"a".repeat(43)}`;
function item(index: number) {
  const digest = index.toString(16).padStart(64, "0");
  const date = new Date(Date.UTC(2026, 9, 3, 12) - index * 1000).toISOString();
  return {
    id: `libentry:sha256:${digest}`, artifact_id: `sha256:${digest}`, version: 1, state: "visible",
    display_name: `Study ${index}`, favorite: false, kind: "image", media_type: "image/png",
    size_bytes: 1024, created_at: date, updated_at: date,
  };
}
function page(indices: number[], next: string | null = null) {
  return parseArtifactLibraryPage({ items: indices.map(item), next_cursor: next }, 20);
}
function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><MediaLibraryView /></QueryClientProvider>);
}

beforeEach(() => {
  vi.resetAllMocks();
  methods.mediaOrganizationCatalog.mockImplementation(async (kind) => ({
    items: kind === "albums" ? [album] : [tag], next_cursor: null, revision: 1,
  }));
  methods.artifactLibrary.mockImplementation(async (filters) => filters.collection_id
    ? parseArtifactLibraryPage({ items: [1, 2].map((index, position) => ({ ...item(index), collection_position: position })), next_cursor: null }, 20, true)
    : page([1, 2]));
  methods.previewMediaOrganization.mockImplementation(async (command) => ({
    id: `orgimp_${"c".repeat(32)}`, action: command.action,
    selected_count: command.entries?.length ?? 0, size_bytes: (command.entries?.length ?? 0) * 1024,
    changed_count: command.entries?.length ?? 1, expires_at: new Date(Date.now() + 900_000).toISOString(),
  }));
  methods.applyMediaOrganization.mockResolvedValue({
    id: `orgimp_${"c".repeat(32)}`, action: "add-to-album", selected_count: 2, changed_count: 2, target_version: 3,
  });
});
afterEach(cleanup);

it.each(["delete-album", "delete-tag", "merge-tags"].flatMap((action) => [true, false].map((affected) => [action, affected] as const)))(
  "clears only the active filter removed by %s when affected is %s", async (action, affected) => {
    let applied = false;
    const destination = { ...tag, id: `mediatag_${"e".repeat(32)}`, label: "Sketch", slug: "sketch" };
    const otherAlbum = { ...album, id: `collection_${"f".repeat(32)}`, name: "Other studies" };
    const activeAlbum = action === "delete-album" && !affected ? otherAlbum.id : album.id;
    const activeTag = action !== "delete-album" && !affected ? destination.id : tag.id;
    methods.mediaOrganizationCatalog.mockImplementation(async (kind) => ({
      items: kind === "albums" ? applied && action === "delete-album" ? [otherAlbum] : [album, otherAlbum]
        : applied && action !== "delete-album" ? [destination] : [tag, destination],
      next_cursor: null, revision: applied ? 2 : 1,
    }));
    methods.artifactLibrary.mockImplementation(async (filters) => {
      if (applied && (action === "delete-album" ? filters.collection_id === album.id : filters.tag_id === tag.id)) {
        throw new ApiError(422, "Start again from the first page.", "Invalid filter", "artifact-library-cursor-invalid");
      }
      return filters.collection_id
        ? parseArtifactLibraryPage({ items: [{ ...item(1), collection_position: 0 }], next_cursor: null }, 20, true) : page([1]);
    });
    methods.previewMediaOrganization.mockResolvedValue({ id: `orgimp_${"c".repeat(32)}`, action,
      selected_count: 0, changed_count: action === "merge-tags" ? 0 : 1, size_bytes: 0,
      expires_at: new Date(Date.now() + 900_000).toISOString() });
    methods.applyMediaOrganization.mockImplementation(async () => {
      applied = true;
      return { id: `orgimp_${"c".repeat(32)}`, action, selected_count: 0,
        changed_count: action === "merge-tags" ? 0 : 1, target_version: null };
    });
    mount();
    await screen.findByRole("option", { name: "Studies" });
    await screen.findByRole("option", { name: "Landscape" });
    fireEvent.change(screen.getByRole("combobox", { name: "Album" }), { target: { value: activeAlbum } });
    fireEvent.change(screen.getByRole("combobox", { name: "Tag" }), { target: { value: activeTag } });
    fireEvent.change(screen.getByRole("textbox", { name: "Search media" }), { target: { value: "Study" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Favorites filter" }), { target: { value: "favorites" } });
    fireEvent.click(screen.getByRole("button", { name: "Albums and tags" }));
    const manager = await screen.findByRole("dialog", { name: "Albums and tags" });
    fireEvent.change(within(manager).getByRole("combobox", { name: "Edit album or tag" }),
      { target: { value: action === "delete-album" ? album.id : tag.id } });
    if (action === "merge-tags") fireEvent.change(within(manager).getByRole("combobox", { name: "Merge into tag" }), { target: { value: destination.id } });
    fireEvent.click(within(manager).getByRole("button", { name: action === "merge-tags" ? "Review tag merge" : "Review removal" }));
    const review = await screen.findByRole("dialog", { name: "Review library changes" });
    expect(screen.getByRole("combobox", { name: action === "delete-album" ? "Album" : "Tag" })).toHaveValue(action === "delete-album" ? activeAlbum : activeTag);
    fireEvent.click(within(review).getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    await waitFor(() => expect(methods.artifactLibrary).toHaveBeenLastCalledWith({
      kind: "", query: "Study", favorite: true,
      collection_id: action === "delete-album" && affected ? undefined : activeAlbum,
      tag_id: action !== "delete-album" && affected ? undefined : activeTag,
    }, null, 20, expect.any(AbortSignal)));
    expect(screen.getByRole("combobox", { name: action === "delete-album" ? "Album" : "Tag" })).toHaveValue(affected ? "" : action === "delete-album" ? activeAlbum : activeTag);
    expect(screen.queryByText("Start again from the first page.")).toBeNull();
    expect(screen.queryByText("The Media Library could not be loaded safely. Refresh and try again.")).toBeNull();
  },
);

it.each([
  ["media-tag-conflict", "This tag name is already in use. Choose a different name.", "Review rename"],
  ["media-tag-merge-trash", "Restore media in Recently Deleted before merging this tag, then review the merge again.", "Review tag merge"],
] as const)("shows an actionable %s refusal inside the editable manager", async (code, message, button) => {
  methods.previewMediaOrganization.mockRejectedValue(new ApiError(409, "neutral raw detail", "Conflict", code));
  const destination = { ...tag, id: `mediatag_${"e".repeat(32)}`, label: "Sketch", slug: "sketch" };
  methods.mediaOrganizationCatalog.mockImplementation(async (kind) => ({
    items: kind === "albums" ? [album] : [tag, destination], next_cursor: null, revision: 1,
  }));
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Albums and tags" }));
  const manager = await screen.findByRole("dialog", { name: "Albums and tags" });
  await within(manager).findByRole("option", { name: "Landscape" });
  fireEvent.change(within(manager).getByRole("combobox", { name: "Edit album or tag" }), { target: { value: tag.id } });
  fireEvent.change(within(manager).getByRole("combobox", { name: "Merge into tag" }), { target: { value: destination.id } });
  fireEvent.click(within(manager).getByRole("button", { name: button }));
  expect(await within(manager).findByText(message)).toBeVisible();
  expect(within(manager).getByRole("textbox", { name: "Tag label" })).not.toHaveAttribute("readonly");
  expect(screen.queryByText("neutral raw detail")).toBeNull();
  expect(methods.applyMediaOrganization).not.toHaveBeenCalled();
});

it("combines album and tag filters without replacing search or favorite filters", async () => {
  mount();
  await screen.findByRole("option", { name: "Studies" });
  await screen.findByRole("option", { name: "Landscape" });
  fireEvent.change(await screen.findByRole("combobox", { name: "Album" }), { target: { value: album.id } });
  fireEvent.change(screen.getByRole("combobox", { name: "Tag" }), { target: { value: tag.id } });
  fireEvent.change(screen.getByRole("textbox", { name: "Search media" }), { target: { value: "Study" } });
  fireEvent.change(screen.getByRole("combobox", { name: "Favorites filter" }), { target: { value: "favorites" } });
  await waitFor(() => expect(methods.artifactLibrary).toHaveBeenLastCalledWith({
    kind: "", query: "Study", favorite: true, collection_id: album.id, tag_id: tag.id,
  }, null, 20, expect.any(AbortSignal)));
});

it("keeps exact selected identities across pages and applies only the confirmed preview", async () => {
  methods.artifactLibrary.mockImplementation(async (_filters, after) => after
    ? page([21]) : page(Array.from({ length: 20 }, (_, index) => index + 1), cursor));
  mount();
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Study 1" }));
  fireEvent.click(screen.getByRole("button", { name: "Load more" }));
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Study 21" }));
  expect(screen.getByText("2 selected · 2 KB")).toBeVisible();
  fireEvent.change(screen.getByRole("combobox", { name: "Selection action" }), { target: { value: "add-to-album" } });
  fireEvent.change(screen.getByRole("combobox", { name: "Destination album" }), { target: { value: album.id } });
  fireEvent.click(screen.getByRole("button", { name: "Review selection" }));
  const dialog = await screen.findByRole("dialog", { name: "Review library changes" });
  expect(methods.previewMediaOrganization).toHaveBeenCalledWith({
    action: "add-to-album", target: { id: album.id, version: 1 },
    entries: [{ id: item(1).id, version: 1 }, { id: item(21).id, version: 1 }],
  });
  expect(methods.applyMediaOrganization).not.toHaveBeenCalled();
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply changes" }));
  await waitFor(() => expect(methods.applyMediaOrganization).toHaveBeenCalledTimes(1));
  expect(methods.applyMediaOrganization).toHaveBeenCalledWith(`orgimp_${"c".repeat(32)}`, expect.any(String));
  await waitFor(() => expect(screen.queryByText("2 selected · 2 KB")).toBeNull());
});

it("keeps the selection when its preview is refused and suppresses raw failure details", async () => {
  methods.previewMediaOrganization.mockRejectedValue(new Error("private stored marker"));
  mount();
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Study 1" }));
  fireEvent.click(screen.getByRole("button", { name: "Review selection" }));
  expect(await screen.findByText("The selected media could not be reviewed. Refresh and select them again.")).toBeVisible();
  expect(screen.getByRole("checkbox", { name: "Select Study 1" })).toBeChecked();
  expect(screen.queryByText("private stored marker")).toBeNull();
  expect(methods.applyMediaOrganization).not.toHaveBeenCalled();
});

it("does not apply a preview whose count disagrees with the selected identities", async () => {
  methods.previewMediaOrganization.mockResolvedValue({
    id: `orgimp_${"c".repeat(32)}`, action: "set-favorite", selected_count: 90,
    changed_count: 1, size_bytes: 1024, expires_at: new Date(Date.now() + 900_000).toISOString(),
  });
  mount();
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Study 1" }));
  fireEvent.click(screen.getByRole("button", { name: "Review selection" }));
  expect(await screen.findByText("The selected media could not be reviewed. Refresh and select them again.")).toBeVisible();
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(methods.applyMediaOrganization).not.toHaveBeenCalled();
});

it("creates an album from the management dialog and refreshes organization choices", async () => {
  methods.createMediaCollection.mockResolvedValue({ ...album, name: "Garden studies" });
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Albums and tags" }));
  const dialog = await screen.findByRole("dialog", { name: "Albums and tags" });
  fireEvent.change(within(dialog).getByRole("textbox", { name: "New album name" }), { target: { value: "Garden studies" } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Create album" }));
  await waitFor(() => expect(methods.createMediaCollection).toHaveBeenCalledWith("Garden studies", "", expect.stringMatching(/^[0-9a-f]{32}$/)));
  await waitFor(() => expect(methods.mediaOrganizationCatalog.mock.calls.length).toBeGreaterThan(2));
});

it("reuses one operation key when retrying an uncertain apply result", async () => {
  methods.applyMediaOrganization.mockRejectedValueOnce(new Error("network unavailable"));
  mount();
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Study 1" }));
  fireEvent.click(screen.getByRole("button", { name: "Review selection" }));
  const dialog = await screen.findByRole("dialog", { name: "Review library changes" });
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply changes" }));
  await screen.findByText("The change could not be confirmed. Try again with this same selection.");
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply changes" }));
  await waitFor(() => expect(methods.applyMediaOrganization).toHaveBeenCalledTimes(2));
  expect(methods.applyMediaOrganization.mock.calls[1]).toEqual(methods.applyMediaOrganization.mock.calls[0]);
});

it("reviews the exact order chosen for selected album members", async () => {
  mount();
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Study 1" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Study 2" }));
  fireEvent.change(screen.getByRole("combobox", { name: "Selection action" }), { target: { value: "reorder-album" } });
  await within(screen.getByRole("combobox", { name: "Destination album" })).findByRole("option", { name: "Studies" });
  fireEvent.change(screen.getByRole("combobox", { name: "Destination album" }), { target: { value: album.id } });
  fireEvent.click(screen.getByRole("button", { name: "Move Study 2 earlier" }));
  fireEvent.click(screen.getByRole("button", { name: "Review selection" }));
  await screen.findByRole("dialog", { name: "Review library changes" });
  expect(methods.previewMediaOrganization).toHaveBeenCalledWith({ action: "reorder-album",
    target: { id: album.id, version: 1 }, entries: [{ id: item(2).id, version: 1 }, { id: item(1).id, version: 1 }] });
  expect(methods.applyMediaOrganization).not.toHaveBeenCalled();
});

it("reviews a tag rename with its exact version and confirms only that change", async () => {
  methods.applyMediaOrganization.mockResolvedValue({ id: `orgimp_${"c".repeat(32)}`, action: "rename-tag",
    selected_count: 0, changed_count: 1, target_version: 2 });
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Albums and tags" }));
  const manager = await screen.findByRole("dialog", { name: "Albums and tags" });
  await within(manager).findByRole("option", { name: "Landscape" });
  fireEvent.change(within(manager).getByRole("combobox", { name: "Edit album or tag" }), { target: { value: tag.id } });
  fireEvent.change(within(manager).getByRole("textbox", { name: "Tag label" }), { target: { value: "Garden studies" } });
  fireEvent.click(within(manager).getByRole("button", { name: "Review rename" }));
  const reviewed = await screen.findByRole("dialog", { name: "Review library changes" });
  expect(within(reviewed).getByText("New name: Garden studies")).toBeVisible();
  expect(methods.previewMediaOrganization).toHaveBeenCalledWith({ action: "rename-tag",
    target: { id: tag.id, version: 1 }, changes: { label: "Garden studies", color: null } });
  expect(methods.applyMediaOrganization).not.toHaveBeenCalled();
  fireEvent.click(within(reviewed).getByRole("button", { name: "Apply changes" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
});

it("cancels album removal without applying its materialized preview", async () => {
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Albums and tags" }));
  const manager = await screen.findByRole("dialog", { name: "Albums and tags" });
  await within(manager).findByRole("option", { name: "Studies" });
  fireEvent.change(within(manager).getByRole("combobox", { name: "Edit album or tag" }), { target: { value: album.id } });
  fireEvent.click(within(manager).getByRole("button", { name: "Review removal" }));
  const reviewed = await screen.findByRole("dialog", { name: "Review library changes" });
  expect(methods.previewMediaOrganization).toHaveBeenCalledWith({ action: "delete-album", target: { id: album.id, version: 1 } });
  expect(within(reviewed).getByText("Media files and existing chats stay intact.")).toBeVisible();
  fireEvent.click(within(reviewed).getByRole("button", { name: "Cancel" }));
  expect(methods.applyMediaOrganization).not.toHaveBeenCalled();
});

it("shows a refused management review inside its dialog without applying it", async () => {
  methods.previewMediaOrganization.mockRejectedValue(new Error("neutral stored marker"));
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Albums and tags" }));
  const manager = await screen.findByRole("dialog", { name: "Albums and tags" });
  await within(manager).findByRole("option", { name: "Studies" });
  fireEvent.change(within(manager).getByRole("combobox", { name: "Edit album or tag" }), { target: { value: album.id } });
  fireEvent.click(within(manager).getByRole("button", { name: "Review removal" }));
  expect(await within(manager).findByText("The selected media could not be reviewed. Refresh and select them again.")).toBeVisible();
  expect(screen.queryByText("neutral stored marker")).toBeNull();
  expect(methods.applyMediaOrganization).not.toHaveBeenCalled();
});

it("merges only the two reviewed tag identities and versions", async () => {
  methods.previewMediaOrganization.mockResolvedValue({ id: `orgimp_${"c".repeat(32)}`,
    action: "merge-tags", selected_count: 0, changed_count: 0, size_bytes: 0,
    expires_at: new Date(Date.now() + 900_000).toISOString() });
  const destination = { ...tag, id: `mediatag_${"e".repeat(32)}`, label: "Sketch", slug: "sketch", version: 4 };
  methods.mediaOrganizationCatalog.mockImplementation(async (kind) => ({
    items: kind === "albums" ? [album] : [tag, destination], next_cursor: null, revision: 1,
  }));
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Albums and tags" }));
  const manager = await screen.findByRole("dialog", { name: "Albums and tags" });
  await within(manager).findByRole("option", { name: "Landscape" });
  fireEvent.change(within(manager).getByRole("combobox", { name: "Edit album or tag" }), { target: { value: tag.id } });
  fireEvent.change(within(manager).getByRole("combobox", { name: "Merge into tag" }), { target: { value: destination.id } });
  fireEvent.click(within(manager).getByRole("button", { name: "Review tag merge" }));
  const reviewed = await screen.findByRole("dialog", { name: "Review library changes" });
  expect(within(reviewed).getByText("Merge into: Sketch")).toBeVisible();
  expect(methods.previewMediaOrganization).toHaveBeenCalledWith({ action: "merge-tags",
    target: { id: tag.id, version: 1 }, destination: { id: destination.id, version: 4 } });
  expect(methods.applyMediaOrganization).not.toHaveBeenCalled();
});

it("loads another bounded catalog page only on request and retains its active filter", async () => {
  const firstAlbums = Array.from({ length: 50 }, (_, index) => ({
    ...album, id: `collection_${index.toString(16).padStart(32, "0")}`, name: `Album ${index}`,
  }));
  methods.mediaOrganizationCatalog.mockImplementation(async (kind, _query, after) => ({
    items: kind === "tags" ? [tag] : after ? [album] : firstAlbums,
    next_cursor: kind === "albums" && !after ? cursor : null, revision: 1,
  }));
  mount();
  await screen.findByRole("option", { name: "Album 0" });
  expect(methods.mediaOrganizationCatalog.mock.calls.filter(([kind]) => kind === "albums")).toHaveLength(1);
  fireEvent.change(screen.getByRole("combobox", { name: "Album" }), { target: { value: firstAlbums[0].id } });
  fireEvent.click(screen.getByText("Find albums and tags"));
  fireEvent.click(screen.getByRole("button", { name: "Next albums" }));
  await screen.findByRole("option", { name: "Studies" });
  expect(screen.getByRole("combobox", { name: "Album" })).toHaveValue(firstAlbums[0].id);
  expect(screen.getByRole("option", { name: "Album 0" })).toBeInTheDocument();
  expect(screen.queryByRole("option", { name: "Album 1" })).toBeNull();
  expect(methods.mediaOrganizationCatalog).toHaveBeenCalledWith("albums", "", cursor, expect.any(AbortSignal));
  fireEvent.click(screen.getByRole("button", { name: "Previous albums" }));
  await screen.findByRole("option", { name: "Album 1" });
});

it("starts catalog search and stale-page recovery from the first page", async () => {
  const firstAlbums = Array.from({ length: 50 }, (_, index) => ({
    ...album, id: `collection_${index.toString(16).padStart(32, "0")}`, name: `Album ${index}`,
  }));
  methods.mediaOrganizationCatalog.mockImplementation(async (kind, query, after) => {
    if (after) throw new Error("neutral stored marker");
    return { items: kind === "tags" ? [tag] : query ? [album] : firstAlbums,
      next_cursor: kind === "albums" && !query ? cursor : null, revision: 1 };
  });
  mount();
  await screen.findByRole("option", { name: "Album 0" });
  fireEvent.click(screen.getByText("Find albums and tags"));
  fireEvent.click(screen.getByRole("button", { name: "Next albums" }));
  await screen.findByText("Albums and tags could not be loaded. Refresh and try again.");
  expect(screen.queryByText("neutral stored marker")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Refresh choices" }));
  await screen.findByRole("option", { name: "Album 0" });
  fireEvent.change(screen.getByRole("textbox", { name: "Search albums" }), { target: { value: " Studies " } });
  await screen.findByRole("option", { name: "Studies" });
  expect(methods.mediaOrganizationCatalog).toHaveBeenCalledWith("albums", "Studies", null, expect.any(AbortSignal));
  expect(screen.queryByRole("option", { name: "Album 0" })).toBeNull();
});

it("keeps the destination name in its review after leaving that catalog page", async () => {
  const different = { ...album, id: `collection_${"e".repeat(32)}`, name: "Different" };
  const firstPage = [album, ...Array.from({ length: 49 }, (_, index) => ({
    ...album, id: `collection_${index.toString(16).padStart(32, "0")}`, name: `Earlier ${index}`,
  }))];
  methods.mediaOrganizationCatalog.mockImplementation(async (kind, _query, after) => ({
    items: kind === "tags" ? [tag] : after ? [different] : firstPage,
    next_cursor: kind === "albums" && !after ? cursor : null, revision: 1,
  }));
  mount();
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Study 1" }));
  fireEvent.change(screen.getByRole("combobox", { name: "Selection action" }), { target: { value: "add-to-album" } });
  const destination = screen.getByRole("combobox", { name: "Destination album" });
  await within(destination).findByRole("option", { name: "Studies" });
  fireEvent.change(destination, { target: { value: album.id } });
  fireEvent.click(screen.getByText("Find albums and tags"));
  fireEvent.click(screen.getByRole("button", { name: "Next albums" }));
  await within(screen.getByRole("combobox", { name: "Album" })).findByRole("option", { name: "Different" });
  fireEvent.click(screen.getByRole("button", { name: "Review selection" }));
  const dialog = await screen.findByRole("dialog", { name: "Review library changes" });
  expect(within(dialog).getByText("Add to album: Studies")).toBeVisible();
  expect(methods.previewMediaOrganization).toHaveBeenCalledWith({ action: "add-to-album",
    target: { id: album.id, version: 1 }, entries: [{ id: item(1).id, version: 1 }] });
});
