import { describe, expect, it } from "vitest";
import {
  insertMention,
  mentionQuery,
  survivingMentions,
  turnReferences,
  type TrackedMention,
} from "./mentionDraft";

const ADA: TrackedMention = { referenceSubjectId: "ref-1", mentionSlug: "ada-lovelace" };
const GRACE: TrackedMention = { referenceSubjectId: "ref-2", mentionSlug: "grace-hopper" };

describe("mentionQuery", () => {
  it("offers everything the moment an @ is typed", () => {
    // Empty is a real answer, not "no query" - the picker should open.
    expect(mentionQuery("draw @", 6)).toBe("");
  });

  it("narrows as the name is typed", () => {
    expect(mentionQuery("draw @ada", 9)).toBe("ada");
  });

  it("says nothing when the caret is not in a mention", () => {
    expect(mentionQuery("draw a picture", 14)).toBeNull();
  });

  it("leaves an email address alone", () => {
    // A mention starts a word. Opening a subject picker while someone types
    // an address would be both wrong and extremely annoying.
    expect(mentionQuery("write to ada@example.com", 24)).toBeNull();
  });

  it("closes once the mention is finished", () => {
    expect(mentionQuery("draw @ada-lovelace holding a cat", 32)).toBeNull();
  });
});

describe("insertMention", () => {
  it("replaces what was typed with the chosen mention", () => {
    const result = insertMention("draw @ad", 8, "ada-lovelace");
    expect(result.text).toBe("draw @ada-lovelace ");
    expect(result.caret).toBe(result.text.length);
  });

  it("keeps whatever followed the caret", () => {
    const result = insertMention("draw @ad holding a cat", 8, "ada-lovelace");
    expect(result.text).toBe("draw @ada-lovelace  holding a cat");
  });

  it("leaves a space so a second mention can follow", () => {
    const first = insertMention("", 0, "ada-lovelace");
    expect(first.text.endsWith(" ")).toBe(true);
    const second = insertMention(first.text, first.caret, "grace-hopper");
    expect(second.text).toBe("@ada-lovelace @grace-hopper ");
  });
});

describe("survivingMentions", () => {
  it("keeps a reference whose mention is still written", () => {
    expect(survivingMentions("draw @ada-lovelace", [ADA])).toEqual([ADA]);
  });

  it("drops a reference whose mention was deleted", () => {
    // Editing the text can remove a reference. That is the only direction it
    // is allowed to work in.
    expect(survivingMentions("draw somebody", [ADA])).toEqual([]);
  });

  it("never promotes typed text into a reference", () => {
    // The characters alone attach nothing. Only the picker attaches an id,
    // because binding whoever the text most resembles is the failure the
    // server refuses and the composer must not reintroduce.
    expect(survivingMentions("draw @grace-hopper", [])).toEqual([]);
  });

  it("does not mistake one subject for another with a longer name", () => {
    const ada: TrackedMention = { referenceSubjectId: "ref-3", mentionSlug: "ada" };
    expect(survivingMentions("draw @ada-lovelace", [ada])).toEqual([]);
  });

  it("keeps only one entry when a subject is mentioned twice", () => {
    const twice = survivingMentions("@ada-lovelace and @ada-lovelace", [ADA, ADA]);
    expect(twice).toEqual([ADA]);
  });

  it("keeps the ones that remain when another is removed", () => {
    expect(survivingMentions("draw @grace-hopper", [ADA, GRACE])).toEqual([GRACE]);
  });
});

describe("turnReferences", () => {
  it("sends ids and how they were chosen, never the text", () => {
    expect(turnReferences([ADA, GRACE])).toEqual([
      { reference_subject_id: "ref-1", source: "mention" },
      { reference_subject_id: "ref-2", source: "mention" },
    ]);
  });

  it("sends nothing when nothing was chosen", () => {
    expect(turnReferences([])).toEqual([]);
  });
});
