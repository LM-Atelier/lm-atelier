import { beforeEach, describe, expect, it } from "vitest";
import { CHAT_ACTIVITY_SEEN_KEY, createChatActivitySeenStore } from "./chatActivitySeen";
import type { ChatActivityReference } from "./types";

const activity = (id: string, sequence: number): ChatActivityReference => ({
  id, sequence, message_id: "message", response_revision_id: "revision",
  occurred_at: "2026-09-20T00:00:00+00:00",
});
beforeEach(() => localStorage.clear());

describe("device-local activity acknowledgment", () => {
  it("marks only the rendered identity, including across a sequence rewind", () => {
    const store = createChatActivitySeenStore(localStorage);
    store.markSeen("chat", activity("newer", 9));
    expect(store.hasSeen("chat", activity("newer", 9))).toBe(true);
    expect(store.hasSeen("chat", activity("older-unseen", 3))).toBe(false);
    expect(store.hasSeen("chat", activity("restored-future", 9))).toBe(false);
    expect(store.hasSeen("other-chat", activity("newer", 9))).toBe(false);
    const reloaded = createChatActivitySeenStore(localStorage);
    expect(reloaded.hasSeen("chat", activity("newer", 9))).toBe(true);
  });

  it("does not infer seen state from opening a chat or a null output", () => {
    const store = createChatActivitySeenStore(localStorage);
    expect(store.hasSeen("chat", null)).toBe(false);
    expect(store.hasSeen("chat", activity("finished-later", 1))).toBe(false);
    expect(localStorage.length).toBe(0);
  });

  it("can keep an explicitly private session entirely in memory", () => {
    const store = createChatActivitySeenStore(null);
    store.markSeen("chat", activity("private", 1));
    expect(store.hasSeen("chat", activity("private", 1))).toBe(true);
    expect(localStorage.length).toBe(0);
    expect(createChatActivitySeenStore(null).hasSeen("chat", activity("private", 1))).toBe(false);
  });

  it("tolerates malformed and denied storage without acknowledging unknown output", () => {
    localStorage.setItem(CHAT_ACTIVITY_SEEN_KEY, "{bad-json");
    expect(createChatActivitySeenStore(localStorage).hasSeen("chat", activity("event", 1))).toBe(false);
    const store = createChatActivitySeenStore({
      getItem: () => { throw new Error("Denied"); },
      setItem: () => { throw new Error("Denied"); },
    });
    store.markSeen("chat", activity("event", 1));
    expect(store.hasSeen("chat", activity("event", 1))).toBe(true);
  });

  it("reconciles cross-tab storage changes and explicit clearing", () => {
    const first = createChatActivitySeenStore(localStorage);
    const second = createChatActivitySeenStore(localStorage);
    first.markSeen("chat", activity("first", 1));
    second.markSeen("chat", activity("second", 2));
    first.refresh(localStorage.getItem(CHAT_ACTIVITY_SEEN_KEY));
    expect(first.hasSeen("chat", activity("first", 1))).toBe(true);
    expect(first.hasSeen("chat", activity("second", 2))).toBe(true);
    first.refresh(null);
    expect(first.hasSeen("chat", activity("first", 1))).toBe(false);
  });

  it("bounds retained identities and expires old browser history", () => {
    let stamp = 1_000;
    const store = createChatActivitySeenStore(localStorage, () => stamp);
    for (let index = 0; index < 2_010; index += 1) {
      stamp += 1;
      store.markSeen("chat", activity(`event-${index}`, index + 1));
    }
    expect(JSON.parse(localStorage.getItem(CHAT_ACTIVITY_SEEN_KEY)!)).toHaveLength(2_000);
    expect(store.hasSeen("chat", activity("event-0", 1))).toBe(false);
    stamp += 91 * 24 * 60 * 60 * 1_000;
    expect(createChatActivitySeenStore(localStorage, () => stamp).hasSeen("chat", activity("event-2009", 2010))).toBe(false);
  });
});
