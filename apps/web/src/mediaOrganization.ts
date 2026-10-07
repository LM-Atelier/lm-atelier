import type { ArtifactLibraryEntry } from "./artifactLibraryPage";

export type OrganizationTarget = { id: string; version: number };
export type MediaAlbum = OrganizationTarget & { kind: "manual"; name: string; description: string };
export type MediaTag = OrganizationTarget & { slug: string; label: string; color: string | null };
export type SelectionAction = "add-to-album" | "remove-from-album" | "reorder-album" | "add-tag" | "remove-tag";
export type OrganizationCommand =
  | { action: SelectionAction; target: OrganizationTarget; entries: OrganizationTarget[] }
  | { action: "set-favorite"; favorite: boolean; entries: OrganizationTarget[] }
  | { action: "trash"; entries: OrganizationTarget[] }
  | { action: "rename-album" | "rename-tag"; target: OrganizationTarget; changes: Record<string, string | null> }
  | { action: "delete-album" | "delete-tag"; target: OrganizationTarget }
  | { action: "merge-tags"; target: OrganizationTarget; destination: OrganizationTarget };
export type OrganizationPreview = {
  id: string; action: OrganizationCommand["action"]; selected_count: number;
  changed_count: number; size_bytes: number; expires_at: string;
};
export const ORGANIZATION_UNAVAILABLE = "The Media Library organization response was invalid.";
function invalid(): never { throw new Error(ORGANIZATION_UNAVAILABLE); }
function object(value: unknown, keys: string[]): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)
    || Object.getPrototypeOf(value) !== Object.prototype || Object.getOwnPropertySymbols(value).length) invalid();
  const actual = Object.keys(value);
  if (actual.length !== keys.length || keys.some((key) => !Object.hasOwn(value, key))) invalid();
  for (const key of actual) if (!Object.hasOwn(Object.getOwnPropertyDescriptor(value, key) ?? {}, "value")) invalid();
  return value as Record<string, unknown>;
}
function count(value: unknown, maximum = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0 || value > maximum) invalid();
  return value;
}
function identity(value: unknown, prefix: string): string {
  if (typeof value !== "string" || !new RegExp(`^${prefix}_[0-9a-f]{32}$`).test(value)) invalid();
  return value;
}
function text(value: unknown, maximum: number, empty = false): string {
  if (typeof value !== "string" || value.length > maximum || (!empty && (!value || value.trim() !== value))
    || /[^ -~]/.test(value)) invalid();
  return value;
}
function version(value: unknown): number { const result = count(value); if (result === 0) invalid(); return result; }
export function parseMediaAlbum(value: unknown): MediaAlbum {
  const row = object(value, ["id", "kind", "name", "description", "version"]);
  if (row.kind !== "manual") invalid();
  return { id: identity(row.id, "collection"), kind: "manual", name: text(row.name, 200),
    description: text(row.description, 2000, true), version: version(row.version) };
}
export function parseMediaTag(value: unknown): MediaTag {
  const row = object(value, ["id", "slug", "label", "color", "version"]);
  const label = text(row.label, 200); const slug = text(row.slug, 80);
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) || slug !== label.toLowerCase().split(/ +/).join("-")) invalid();
  if (row.color !== null && (typeof row.color !== "string" || !/^#[0-9a-f]{6}$/.test(row.color))) invalid();
  return { id: identity(row.id, "mediatag"), slug, label, color: row.color, version: version(row.version) };
}
export function parseMediaCatalog<T extends { id: string }>(value: unknown, key: string, parse: (row: unknown) => T): T[] {
  const row = object(value, [key]); const items = row[key];
  if (!Array.isArray(items) || items.length > 100_000) invalid();
  const seen = new Set<string>();
  return items.map((item) => {
    const parsed = parse(item);
    if (seen.has(parsed.id)) invalid(); seen.add(parsed.id); return parsed;
  });
}
export type MediaCatalogPage<T> = { items: T[]; next_cursor: string | null; revision: number };
export function parseMediaCatalogPage<T extends { id: string }>(
  value: unknown, parse: (row: unknown) => T, limit = 50,
): MediaCatalogPage<T> {
  const row = object(value, ["items", "next_cursor", "revision"]);
  if (!Array.isArray(row.items) || row.items.length > limit
    || (row.next_cursor !== null && (typeof row.next_cursor !== "string"
      || row.next_cursor.length > 2048 || !/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{43}$/.test(row.next_cursor)
      || row.items.length !== limit))) invalid();
  const items = parseMediaCatalog({ items: row.items }, "items", parse);
  return { items, next_cursor: row.next_cursor, revision: version(row.revision) };
}
export function parseOrganizationPreview(
  value: unknown, command: OrganizationCommand, selection?: ArtifactLibraryEntry[],
): OrganizationPreview {
  const row = object(value, ["id", "action", "selected_count", "changed_count", "size_bytes", "expires_at"]);
  const selected = count(row.selected_count, 100_000);
  const ceiling = ["rename-album", "rename-tag", "delete-album", "delete-tag"].includes(command.action)
    ? Math.max(1, selected) : selected;
  const changed = count(row.changed_count, ceiling);
  const size = count(row.size_bytes);
  if (row.action !== command.action || typeof row.expires_at !== "string"
    || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)$/.test(row.expires_at)
    || !Number.isFinite(Date.parse(row.expires_at))
    || new Date(row.expires_at).toISOString().slice(0, 19) !== row.expires_at.slice(0, 19)) invalid();
  if ("entries" in command && selected !== command.entries.length) invalid();
  if (selection && (selected !== selection.length || size !== selection.reduce((sum, item) => sum + item.size_bytes, 0))) invalid();
  return { id: identity(row.id, "orgimp"), action: command.action, selected_count: selected,
    changed_count: changed, size_bytes: size, expires_at: row.expires_at };
}
export function parseOrganizationResult(value: unknown, preview: OrganizationPreview, command: OrganizationCommand): void {
  const keys = ["id", "action", "selected_count", "changed_count", "target_version"];
  if (command.action === "trash") keys.push("deletion_ids");
  const row = object(value, keys);
  const hasVersion = "target" in command && !["delete-album", "delete-tag", "merge-tags"].includes(command.action);
  let expected: number | null = null;
  if (hasVersion && "target" in command) expected = command.target.version + (command.action === "reorder-album"
    ? preview.changed_count ? preview.selected_count * 2 : 0 : preview.changed_count);
  if (expected !== null && !Number.isSafeInteger(expected)) invalid();
  if (row.id !== preview.id || row.action !== preview.action || row.selected_count !== preview.selected_count
    || row.changed_count !== preview.changed_count || row.target_version !== expected) invalid();
  if (command.action === "trash") {
    const ids = row.deletion_ids;
    if (!Array.isArray(ids) || ids.length !== preview.selected_count || new Set(ids).size !== ids.length) invalid();
    ids.forEach((id) => identity(id, "recover"));
  }
}
export function organizationOperationKey(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
