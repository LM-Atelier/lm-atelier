import { expect, it } from "vitest";
import {
  ORGANIZATION_UNAVAILABLE, parseMediaAlbum, parseMediaCatalog, parseMediaCatalogPage, parseMediaTag,
  parseOrganizationPreview, parseOrganizationResult, type OrganizationCommand,
} from "./mediaOrganization";

const album = { id: `collection_${"a".repeat(32)}`, kind: "manual", name: "Studies", description: "", version: 1 };
const tag = { id: `mediatag_${"b".repeat(32)}`, slug: "garden-study", label: "Garden   Study", color: "#aabbcc", version: 2 };
const command: OrganizationCommand = { action: "add-to-album", target: { id: album.id, version: 1 },
  entries: [{ id: `libentry:sha256:${"c".repeat(64)}`, version: 1 }] };
const preview = { id: `orgimp_${"d".repeat(32)}`, action: "add-to-album" as const, selected_count: 1,
  changed_count: 1, size_bytes: 1024, expires_at: "2026-10-03T22:00:00.000000+00:00" };
const result = { id: preview.id, action: preview.action, selected_count: 1, changed_count: 1, target_version: 2 };

it("accepts one bounded catalog page and its exact revision", () => {
  expect(parseMediaCatalogPage({ items: [album], next_cursor: null, revision: 4 }, parseMediaAlbum)).toEqual({
    items: [album], next_cursor: null, revision: 4,
  });
});
it.each([
  { items: [album], next_cursor: null, revision: true },
  { items: [album], next_cursor: null, revision: 0 },
  { items: [album, album], next_cursor: null, revision: 1 },
  { items: [album], next_cursor: "invalid", revision: 1 },
  { items: [album], next_cursor: `cGF5bG9hZA.${"a".repeat(43)}`, revision: 1 },
  { items: Array.from({ length: 51 }, () => album), next_cursor: null, revision: 1 },
  { items: [album], next_cursor: null, revision: 1, unexpected: "neutral marker" },
])("refuses a damaged, oversized or unbound catalog page", (value) => {
  expect(() => parseMediaCatalogPage(value, parseMediaAlbum)).toThrow(ORGANIZATION_UNAVAILABLE);
});

it("accepts normalized names and exact catalog identities", () => {
  expect(parseMediaAlbum(album)).toEqual(album);
  expect(parseMediaTag(tag)).toEqual(tag);
  expect(parseMediaCatalog({ collections: [album] }, "collections", parseMediaAlbum)).toEqual([album]);
  expect(parseOrganizationPreview(preview, command)).toEqual(preview);
  expect(() => parseOrganizationResult(result, preview, command)).not.toThrow();
});
it.each([
  { ...album, version: true }, { ...album, id: "invalid" }, { ...album, unexpected: "neutral stored marker" },
])("refuses damaged catalog records without returning their fields", (value) => {
  expect(() => parseMediaAlbum(value)).toThrow(ORGANIZATION_UNAVAILABLE);
});
it("refuses duplicate IDs and inconsistent normalized tag names", () => {
  expect(() => parseMediaCatalog({ collections: [album, album] }, "collections", parseMediaAlbum)).toThrow(ORGANIZATION_UNAVAILABLE);
  expect(() => parseMediaTag({ ...tag, slug: "wrong-name" })).toThrow(ORGANIZATION_UNAVAILABLE);
});
it("refuses a getter without evaluating it", () => {
  let evaluated = false;
  const value = { ...album };
  Object.defineProperty(value, "name", { enumerable: true, get: () => { evaluated = true; return "neutral stored marker"; } });
  expect(() => parseMediaAlbum(value)).toThrow(ORGANIZATION_UNAVAILABLE);
  expect(evaluated).toBe(false);
});
it.each([
  { ...preview, selected_count: 2 }, { ...preview, selected_count: true },
  { ...preview, changed_count: 2 }, { ...preview, size_bytes: Number.NaN },
  { ...preview, expires_at: "2026-02-31T22:00:00Z" }, { ...preview, expires_at: "2026-10-03T22:00:00" },
  { ...preview, action: "trash" }, { ...preview, unexpected: "neutral stored marker" },
])("refuses an unbound or malformed preview", (value) => {
  expect(() => parseOrganizationPreview(value, command)).toThrow(ORGANIZATION_UNAVAILABLE);
});
it.each([
  { ...result, target_version: 9 }, { ...result, changed_count: false }, { ...result, selected_count: 2 },
  { ...result, action: "trash" }, { ...result, unexpected: "neutral stored marker" },
  { ...result, deletion_ids: ["recover_" + "e".repeat(32)] },
])("refuses a result that cannot confirm the reviewed change", (value) => {
  expect(() => parseOrganizationResult(value, preview, command)).toThrow(ORGANIZATION_UNAVAILABLE);
});
it("refuses a repeated deletion identity in a Trash result", () => {
  const trash: OrganizationCommand = { action: "trash", entries: [...command.entries, { id: `libentry:sha256:${"f".repeat(64)}`, version: 1 }] };
  const reviewed = { ...preview, action: "trash" as const, selected_count: 2, changed_count: 2 };
  expect(() => parseOrganizationResult({ ...result, action: "trash", selected_count: 2, changed_count: 2,
    target_version: null, deletion_ids: ["recover_" + "e".repeat(32), "recover_" + "e".repeat(32)] }, reviewed, trash)).toThrow(ORGANIZATION_UNAVAILABLE);
});
