/** The Studio's unfinished work, kept in this browser so a reload or a restart does not lose it.
 *
 * The kept draft lasts as long as the app is open. This is the same draft
 * written down when the Studio is left or the page is hidden, and given back
 * once, on the same picture of the same session. It keeps what takes effort to
 * make again: the words, the words the text tools were given, the tool in
 * hand and the selection. The tools' sizes and sliders start from their
 * defaults. The selection is stored as its runs, and left out when even those
 * are too large to keep here.
 */

import { createMask, decodeMask, encodeMask, type MaskRaster } from "./studioMasks";
import { initialToolState, SELECTION_KINDS, type StudioToolState } from "./studioToolState";
import type { StudioToolKind } from "./types";

export const SAVED_DRAFT_KEY = "local-lm-studio-saved-draft";

/** The largest selection kept, in stored bytes; a larger one is left behind rather than filling the browser's storage. */
export const MAX_SAVED_SELECTION_BYTES = 2 * 1024 * 1024;

export type SavedStudioDraft = {
  sessionId: string;
  artifactId: string;
  instruction: string;
  selectedId: string | null;
  tools: StudioToolState;
};

/** Every tool, so a stored name from another version of the Studio is never put in hand. */
const TOOL_KINDS: Record<StudioToolKind, true> = {
  instruct: true, brush: true, eraser: true, rect: true, lasso: true, bucket: true, wand: true, enhance: true,
  extend: true, text: true, remove: true, relight: true, isolate: true, background: true, subject: true,
  transform: true, perspective: true, crop: true, resize: true, adjust: true, blur: true, paint: true,
  caption: true, canvas: true,
};

type Stored = {
  version: 1;
  sessionId: string;
  artifactId: string;
  instruction: string;
  selectedId: string | null;
  kind: string;
  selectionKind: string;
  currentWords: string;
  newWords: string;
  caption: string;
  /** The selection's size always, and its runs or bytes when they are small enough to keep. */
  selection: { width: number; height: number; runs?: string; bytes?: string } | null;
};

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let start = 0; start < bytes.length; start += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(start, start + 0x8000));
  }
  return btoa(binary);
}

function fromBase64(text: string): Uint8Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function storedSelection(mask: MaskRaster | null): Stored["selection"] {
  if (!mask) return null;
  const snapshot = encodeMask(mask);
  const raw = "bytes" in snapshot
    ? snapshot.bytes
    : new Uint8Array(snapshot.runs.buffer, snapshot.runs.byteOffset, snapshot.runs.byteLength);
  if (raw.byteLength > MAX_SAVED_SELECTION_BYTES) return { width: snapshot.width, height: snapshot.height };
  const encoded = toBase64(raw);
  return "bytes" in snapshot
    ? { width: snapshot.width, height: snapshot.height, bytes: encoded }
    : { width: snapshot.width, height: snapshot.height, runs: encoded };
}

/** Write the draft down, replacing whatever was kept before. */
export function saveStudioDraft(draft: SavedStudioDraft): void {
  const { tools } = draft;
  const stored: Stored = {
    version: 1,
    sessionId: draft.sessionId,
    artifactId: draft.artifactId,
    instruction: draft.instruction,
    selectedId: draft.selectedId,
    kind: tools.kind,
    selectionKind: tools.selectionKind,
    currentWords: tools.currentWords,
    newWords: tools.newWords,
    caption: tools.caption.text,
    selection: storedSelection(tools.mask),
  };
  try {
    localStorage.setItem(SAVED_DRAFT_KEY, JSON.stringify(stored));
  } catch {
    // Storage that is full or unavailable keeps nothing; the draft in memory still lasts the visit.
  }
}

function readStored(): Stored | null {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(SAVED_DRAFT_KEY) ?? "null");
    if (typeof parsed !== "object" || parsed === null) return null;
    const value = parsed as Partial<Stored>;
    const texts = [value.sessionId, value.artifactId, value.instruction, value.kind, value.selectionKind,
      value.currentWords, value.newWords, value.caption];
    if (value.version !== 1 || !texts.every((text) => typeof text === "string")) return null;
    if (value.selectedId !== null && typeof value.selectedId !== "string") return null;
    return value as Stored;
  } catch {
    return null;
  }
}

/** The stored selection, when it is whole and drawn at the picture's size; anything else is none. */
function restoredSelection(selection: Stored["selection"], size: { width: number; height: number }): MaskRaster | null {
  if (!selection || selection.width !== size.width || selection.height !== size.height) return null;
  try {
    if (typeof selection.bytes === "string") {
      const bytes = fromBase64(selection.bytes);
      if (bytes.length !== selection.width * selection.height) return null;
      return decodeMask({ width: selection.width, height: selection.height, bytes });
    }
    if (typeof selection.runs !== "string") return null;
    const raw = fromBase64(selection.runs);
    if (raw.byteLength % 8 !== 0) return null;
    const runs = new Uint32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    let covered = 0;
    for (let index = 1; index < runs.length; index += 2) covered += runs[index];
    if (covered !== selection.width * selection.height) return null;
    return decodeMask({ width: selection.width, height: selection.height, runs });
  } catch {
    return null;
  }
}

/** The draft written down for this session, if any: its words and chosen step come back with the session. */
export function savedStudioDraftWords(sessionId: string | null): { instruction: string; selectedId: string | null } | null {
  const stored = sessionId ? readStored() : null;
  return stored && stored.sessionId === sessionId
    ? { instruction: stored.instruction, selectedId: stored.selectedId }
    : null;
}

/** The draft written down for this picture of this session, given back once, its selection fitted to the picture. */
export function takeSavedStudioDraft(
  sessionId: string | null,
  artifactId: string | null,
  size: { width: number; height: number },
): SavedStudioDraft | null {
  const stored = readStored();
  if (!stored || !sessionId || stored.sessionId !== sessionId || stored.artifactId !== artifactId) return null;
  try {
    localStorage.removeItem(SAVED_DRAFT_KEY);
  } catch {
    // Given back all the same; a later save replaces it.
  }
  const initial = initialToolState();
  const selectionKind = (SELECTION_KINDS as readonly string[]).includes(stored.selectionKind)
    ? stored.selectionKind as StudioToolState["selectionKind"]
    : initial.selectionKind;
  const tools: StudioToolState = {
    ...initial,
    kind: Object.hasOwn(TOOL_KINDS, stored.kind) ? stored.kind as StudioToolKind : initial.kind,
    selectionKind,
    currentWords: stored.currentWords,
    newWords: stored.newWords,
    caption: { ...initial.caption, text: stored.caption },
    mask: restoredSelection(stored.selection, size) ?? createMask(size.width, size.height),
    maskVersion: 1,
  };
  return {
    sessionId: stored.sessionId,
    artifactId: stored.artifactId,
    instruction: stored.instruction,
    selectedId: stored.selectedId,
    tools,
  };
}
