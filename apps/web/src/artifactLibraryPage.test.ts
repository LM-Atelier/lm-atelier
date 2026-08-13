import { describe, expect, it } from "vitest";
import {
  ARTIFACT_LIBRARY_PAGE_ERROR,
  flattenArtifactLibraryPages,
  parseArtifactLibraryPage,
} from "./artifactLibraryPage";

const digest = (character: string) => character.repeat(64);
const cursor = `cGF5bG9hZA.${"a".repeat(43)}`;

function item(character = "a", createdAt = "2026-08-12T12:00:00Z") {
  const sha = digest(character);
  return {
    id: `libentry:sha256:${sha}`,
    artifact_id: `sha256:${sha}`,
    version: 1,
    state: "visible",
    display_name: `Item ${character}`,
    favorite: false,
    kind: "image",
    media_type: "image/png",
    size_bytes: 42,
    created_at: createdAt,
    updated_at: createdAt,
  };
}

const page = (items: unknown[], nextCursor: unknown = null) => ({
  items,
  next_cursor: nextCursor,
});

function rejects(value: unknown, limit = 20) {
  expect(() => parseArtifactLibraryPage(value, limit)).toThrow(ARTIFACT_LIBRARY_PAGE_ERROR);
}

describe("Media Library EntryV1 response boundary", () => {
  it("reconstructs the exact safe page", () => {
    const parsed = parseArtifactLibraryPage(page([item()]), 20);
    expect(parsed.items[0]).toMatchObject({
      artifact_id: `sha256:${digest("a")}`,
      display_name: "Item a",
      created_at_epoch_micros: Date.parse("2026-08-12T12:00:00Z") * 1000,
    });
    expect(parsed.next_cursor).toBeNull();
  });

  it("refuses malformed identities, scalars, media, times, and private extras", () => {
    const base = item();
    const mutations: unknown[] = [
      { ...base, id: `libentry:sha256:${digest("b")}` },
      { ...base, artifact_id: "artifact-1" },
      { ...base, version: true },
      { ...base, version: Number.MAX_SAFE_INTEGER + 1 },
      { ...base, state: "trashed" },
      { ...base, favorite: "false" },
      { ...base, kind: "other" },
      { ...base, media_type: " image/png" },
      { ...base, media_type: "image/\u007fprivate" },
      { ...base, display_name: "private\u0000name" },
      { ...base, size_bytes: 0 },
      { ...base, created_at: "2026-02-30T00:00:00Z" },
      { ...base, created_at: "2026-08-12T00:00:00" , updated_at: "2026-08-11T00:00:00Z" },
      { ...base, relative_path: "private/path" },
    ];
    for (const mutation of mutations) rejects(page([mutation]));
  });

  it("refuses wrong page shapes, duplicates, oversized pages, and impossible cursors", () => {
    rejects({ items: [], next_cursor: null, total: 0 });
    rejects(page("not-a-list" as never));
    rejects(page([item(), item()]));
    rejects(page([item()], "not.a.cursor"), 1);
    rejects(page([], "opaque"), 1);
    rejects(page([item()], "opaque"), 20);
    rejects(page([item(), item("b")]), 1);
  });

  it("accepts the backend cursor maximum and refuses segment overflow", () => {
    const maximumCursor = `${"a".repeat(1600)}.${"b".repeat(43)}`;
    expect(parseArtifactLibraryPage(page([item()], maximumCursor), 1).next_cursor)
      .toBe(maximumCursor);
    rejects(page([item()], `${"a".repeat(1601)}.${"b".repeat(43)}`), 1);
    rejects(page([item()], `${"a".repeat(1600)}.${"b".repeat(44)}`), 1);
  });

  it("counts display-name bounds by Unicode code point like the backend", () => {
    const maximum = "😀".repeat(500);
    expect(parseArtifactLibraryPage(
      page([{ ...item(), display_name: maximum }]),
      20,
    ).items[0].display_name).toBe(maximum);
    rejects(page([{ ...item(), display_name: "😀".repeat(501) }]));
  });

  it("is total for hostile non-JSON objects", () => {
    const accessor = Object.create(Object.prototype, {
      items: { get: () => { throw new Error("private marker"); }, enumerable: true },
      next_cursor: { value: null, enumerable: true },
    });
    rejects(accessor);
    rejects(Object.create({ items: [], next_cursor: null }));
    rejects(new (class Page { items = []; next_cursor = null; })());
  });
});

describe("cursor-chain validation", () => {
  it("accepts strict descending time and id order", () => {
    const first = parseArtifactLibraryPage(
      page([item("b"), item("a")], cursor),
      2,
    );
    const second = parseArtifactLibraryPage(
      page([item("c", "2026-08-12T11:59:00Z")]),
      2,
    );
    expect(flattenArtifactLibraryPages([first, second])).toHaveLength(3);
  });

  it("refuses duplicates, order reversals, and pages after terminal", () => {
    const terminal = parseArtifactLibraryPage(page([item("a")]), 2);
    const later = parseArtifactLibraryPage(page([item("b", "2026-08-12T12:01:00Z")]), 2);
    expect(() => flattenArtifactLibraryPages([terminal, later])).toThrow(ARTIFACT_LIBRARY_PAGE_ERROR);

    const first = parseArtifactLibraryPage(page([item("b"), item("a")], cursor), 2);
    const duplicate = parseArtifactLibraryPage(page([item("a")]), 2);
    expect(() => flattenArtifactLibraryPages([first, duplicate])).toThrow(ARTIFACT_LIBRARY_PAGE_ERROR);

    const reversed = parseArtifactLibraryPage(page([item("c", "2026-08-12T12:01:00Z")]), 2);
    expect(() => flattenArtifactLibraryPages([first, reversed])).toThrow(ARTIFACT_LIBRARY_PAGE_ERROR);
  });

  it("preserves microsecond ordering that Date.parse alone would collapse", () => {
    const first = parseArtifactLibraryPage(
      page([item("b", "2026-08-12T12:00:00.000002Z"), item("a", "2026-08-12T12:00:00.000001Z")], cursor),
      2,
    );
    expect(flattenArtifactLibraryPages([first])).toHaveLength(2);
    const reversed = parseArtifactLibraryPage(
      page([item("a", "2026-08-12T12:00:00.000001Z"), item("b", "2026-08-12T12:00:00.000002Z")], cursor),
      2,
    );
    expect(() => flattenArtifactLibraryPages([reversed])).toThrow(ARTIFACT_LIBRARY_PAGE_ERROR);
  });
});
