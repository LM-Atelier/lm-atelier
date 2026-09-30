/** The Studio draft written down in this browser, so a reload gives back the words and the selection. */

import { describe, expect, it } from "vitest";
import { coverage, createMask, fillRect, isEmpty } from "./studioMasks";
import {
  MAX_SAVED_SELECTION_BYTES,
  SAVED_DRAFT_KEY,
  saveStudioDraft,
  savedStudioDraftWords,
  takeSavedStudioDraft,
} from "./studioSavedDraft";
import { initialToolState, type StudioToolState } from "./studioToolState";

const SIZE = { width: 40, height: 20 };

function drawn(tools: Partial<StudioToolState> = {}): StudioToolState {
  const mask = createMask(SIZE.width, SIZE.height);
  fillRect(mask, 0, 0, 20, 20);
  return { ...initialToolState(), kind: "brush", mask, maskVersion: 3, ...tools };
}

function save(tools: StudioToolState, sessionId = "chat-studio", artifactId = "art-1") {
  saveStudioDraft({ sessionId, artifactId, instruction: "make the sky warmer", selectedId: "art-1", tools });
}

describe("the draft written down in this browser", () => {
  it("gives back the words, the tool, the text tools' words and the selection, once", () => {
    save(drawn({ selectionKind: "lasso", currentWords: "OPEN", newWords: "CLOSED", caption: { ...initialToolState().caption, text: "Harbour" } }));

    expect(savedStudioDraftWords("chat-studio")).toEqual({ instruction: "make the sky warmer", selectedId: "art-1" });
    const draft = takeSavedStudioDraft("chat-studio", "art-1", SIZE)!;

    expect(draft.instruction).toBe("make the sky warmer");
    expect(draft.selectedId).toBe("art-1");
    expect(draft.tools.kind).toBe("brush");
    expect(draft.tools.selectionKind).toBe("lasso");
    expect([draft.tools.currentWords, draft.tools.newWords, draft.tools.caption.text]).toEqual(["OPEN", "CLOSED", "Harbour"]);
    expect(coverage(draft.tools.mask!)).toBeCloseTo(0.5);
    // Sizes and sliders start from their defaults, and nothing is left to undo.
    expect(draft.tools.brushRadius).toBe(initialToolState().brushRadius);
    expect(draft.tools.history.canUndo).toBe(false);
    expect(takeSavedStudioDraft("chat-studio", "art-1", SIZE)).toBeNull();
  });

  it("is kept for its own picture of its own session only", () => {
    save(drawn());

    expect(savedStudioDraftWords("chat-other")).toBeNull();
    expect(takeSavedStudioDraft("chat-other", "art-1", SIZE)).toBeNull();
    expect(takeSavedStudioDraft("chat-studio", "art-2", SIZE)).toBeNull();
    // Still there for the picture it was made on.
    expect(takeSavedStudioDraft("chat-studio", "art-1", SIZE)).not.toBeNull();
  });

  it("keeps the words but not a selection too large to keep, or one of another size", () => {
    // A speckled selection stores byte for byte, and this one is larger than the limit.
    const side = Math.ceil(Math.sqrt(MAX_SAVED_SELECTION_BYTES)) + 1;
    const speckled = createMask(side, side);
    for (let index = 0; index < speckled.data.length; index += 2) speckled.data[index] = 255;
    save({ ...drawn(), mask: speckled });

    const large = takeSavedStudioDraft("chat-studio", "art-1", { width: side, height: side })!;
    expect(large.instruction).toBe("make the sky warmer");
    expect(isEmpty(large.tools.mask!)).toBe(true);
    expect([large.tools.mask!.width, large.tools.mask!.height]).toEqual([side, side]);

    save(drawn());
    const elsewhere = takeSavedStudioDraft("chat-studio", "art-1", { width: 80, height: 40 })!;
    expect(isEmpty(elsewhere.tools.mask!)).toBe(true);
    expect([elsewhere.tools.mask!.width, elsewhere.tools.mask!.height]).toEqual([80, 40]);
  });

  it("puts no unknown tool in hand, and reads nothing from what it did not write", () => {
    save(drawn());
    const stored = JSON.parse(localStorage.getItem(SAVED_DRAFT_KEY)!);
    // A name another version might write, and one every object carries.
    localStorage.setItem(SAVED_DRAFT_KEY, JSON.stringify({ ...stored, kind: "constructor", selectionKind: "toString" }));
    const draft = takeSavedStudioDraft("chat-studio", "art-1", SIZE)!;
    expect(draft.tools.kind).toBe(initialToolState().kind);
    expect(draft.tools.selectionKind).toBe(initialToolState().selectionKind);

    for (const unreadable of ["not json", JSON.stringify({ ...stored, version: 2 }), JSON.stringify({ ...stored, instruction: 7 })]) {
      localStorage.setItem(SAVED_DRAFT_KEY, unreadable);
      expect(savedStudioDraftWords("chat-studio")).toBeNull();
      expect(takeSavedStudioDraft("chat-studio", "art-1", SIZE)).toBeNull();
    }

    // Runs that do not cover the picture give the words back with an empty selection.
    localStorage.setItem(SAVED_DRAFT_KEY, JSON.stringify({ ...stored, selection: { width: 40, height: 20, runs: "AAAA" } }));
    const broken = takeSavedStudioDraft("chat-studio", "art-1", SIZE)!;
    expect(broken.instruction).toBe("make the sky warmer");
    expect(isEmpty(broken.tools.mask!)).toBe(true);
  });
});
