const PAGE_KEYS = ["items", "next_cursor"] as const;
const ITEM_KEYS = [
  "id",
  "artifact_id",
  "version",
  "state",
  "display_name",
  "favorite",
  "kind",
  "media_type",
  "size_bytes",
  "created_at",
  "updated_at",
] as const;

const DIGEST = /^[0-9a-f]{64}$/;
const CURSOR = /^[A-Za-z0-9_-]{1,1600}\.[A-Za-z0-9_-]{43}$/;
const TIMESTAMP = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(Z|[+-]\d{2}:\d{2})?$/;

export const ARTIFACT_LIBRARY_PAGE_ERROR = "The Media Library response was invalid.";

export type ArtifactLibraryKind = "image" | "video";

export interface ArtifactLibraryEntry {
  id: string;
  artifact_id: string;
  version: number;
  state: "visible";
  display_name: string;
  favorite: boolean;
  kind: ArtifactLibraryKind;
  media_type: string;
  size_bytes: number;
  created_at: string;
  updated_at: string;
  /** Parsed once at the trust boundary; used only for exact page ordering. */
  created_at_epoch_micros: number;
}

export interface ArtifactLibraryPage {
  items: ArtifactLibraryEntry[];
  next_cursor: string | null;
}

export interface ArtifactLibraryFilters {
  kind: "" | ArtifactLibraryKind;
  query: string;
  favorite: boolean;
}

function invalid(): never {
  throw new Error(ARTIFACT_LIBRARY_PAGE_ERROR);
}

function plainDataObject(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) invalid();
  let prototype: object | null;
  let keys: string[];
  let symbols: symbol[];
  try {
    prototype = Object.getPrototypeOf(value);
    keys = Object.keys(value);
    symbols = Object.getOwnPropertySymbols(value);
  } catch {
    return invalid();
  }
  if (prototype !== Object.prototype || symbols.length !== 0) invalid();
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !("value" in descriptor)) invalid();
  }
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[]): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    invalid();
  }
}

function boundedString(value: unknown, minimum: number, maximum: number): string {
  if (typeof value !== "string" || value.length < minimum || value.length > maximum) invalid();
  for (const character of value) {
    const code = character.charCodeAt(0);
    if (code < 32 || code === 127) invalid();
  }
  return value;
}

function positiveSafeInteger(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) invalid();
  return value;
}

function timestamp(value: unknown): { text: string; epochMicros: number } {
  const text = boundedString(value, 19, 32);
  const match = TIMESTAMP.exec(text);
  if (!match) invalid();
  const [, yearText, monthText, dayText, hourText, minuteText, secondText, fraction = "", zone = ""] = match;
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const hour = Number(hourText);
  const minute = Number(minuteText);
  const second = Number(secondText);
  if (year < 1970 || month < 1 || month > 12 || hour > 23 || minute > 59 || second > 59) invalid();
  const days = new Date(Date.UTC(year, month, 0)).getUTCDate();
  if (day < 1 || day > days) invalid();
  if (zone && zone !== "Z") {
    const zoneHour = Number(zone.slice(1, 3));
    const zoneMinute = Number(zone.slice(4, 6));
    if (zoneHour > 23 || zoneMinute > 59) invalid();
  }
  const normalized = `${yearText}-${monthText}-${dayText}T${hourText}:${minuteText}:${secondText}${fraction ? `.${fraction}` : ""}${zone || "Z"}`;
  const epochMs = Date.parse(normalized);
  if (!Number.isFinite(epochMs)) invalid();
  const fractionMicros = Number(`${fraction}000000`.slice(0, 6));
  const epochMicros = epochMs * 1000 + (fractionMicros % 1000);
  if (!Number.isSafeInteger(epochMicros)) invalid();
  return { text, epochMicros };
}

function parseEntry(value: unknown): ArtifactLibraryEntry {
  const record = plainDataObject(value);
  exactKeys(record, ITEM_KEYS);
  const id = boundedString(record.id, 80, 80);
  const artifactId = boundedString(record.artifact_id, 71, 71);
  const entryMatch = /^libentry:sha256:([0-9a-f]{64})$/.exec(id);
  const artifactMatch = /^sha256:([0-9a-f]{64})$/.exec(artifactId);
  if (!entryMatch || !artifactMatch || !DIGEST.test(entryMatch[1]) || entryMatch[1] !== artifactMatch[1]) invalid();
  if (record.state !== "visible" || typeof record.favorite !== "boolean") invalid();
  if (record.kind !== "image" && record.kind !== "video") invalid();
  const mediaType = boundedString(record.media_type, 1, 120);
  if (mediaType !== mediaType.trim()) invalid();
  const created = timestamp(record.created_at);
  const updated = timestamp(record.updated_at);
  if (updated.epochMicros < created.epochMicros) invalid();
  return Object.freeze({
    id,
    artifact_id: artifactId,
    version: positiveSafeInteger(record.version),
    state: "visible" as const,
    display_name: boundedString(record.display_name, 1, 500),
    favorite: record.favorite,
    kind: record.kind,
    media_type: mediaType,
    size_bytes: positiveSafeInteger(record.size_bytes),
    created_at: created.text,
    updated_at: updated.text,
    created_at_epoch_micros: created.epochMicros,
  });
}

export function parseArtifactLibraryPage(value: unknown, requestedLimit: number): ArtifactLibraryPage {
  if (!Number.isSafeInteger(requestedLimit) || requestedLimit < 1 || requestedLimit > 100) invalid();
  const record = plainDataObject(value);
  exactKeys(record, PAGE_KEYS);
  if (!Array.isArray(record.items) || record.items.length > requestedLimit) invalid();
  const items = record.items.map(parseEntry);
  const ids = new Set(items.map((item) => item.id));
  if (ids.size !== items.length) invalid();
  const nextCursor = record.next_cursor === null
    ? null
    : boundedString(record.next_cursor, 1, 1600);
  if (nextCursor !== null && (!CURSOR.test(nextCursor) || items.length !== requestedLimit)) invalid();
  return Object.freeze({ items: Object.freeze(items) as ArtifactLibraryEntry[], next_cursor: nextCursor });
}

function comesBefore(left: ArtifactLibraryEntry, right: ArtifactLibraryEntry): boolean {
  return left.created_at_epoch_micros > right.created_at_epoch_micros
    || (left.created_at_epoch_micros === right.created_at_epoch_micros && left.id > right.id);
}

export function flattenArtifactLibraryPages(pages: readonly ArtifactLibraryPage[]): ArtifactLibraryEntry[] {
  const result: ArtifactLibraryEntry[] = [];
  const ids = new Set<string>();
  let previous: ArtifactLibraryEntry | undefined;
  for (let pageIndex = 0; pageIndex < pages.length; pageIndex += 1) {
    const page = pages[pageIndex];
    if (pageIndex < pages.length - 1 && page.next_cursor === null) invalid();
    for (const item of page.items) {
      if (ids.has(item.id) || (previous && !comesBefore(previous, item))) invalid();
      ids.add(item.id);
      result.push(item);
      previous = item;
    }
  }
  return result;
}
