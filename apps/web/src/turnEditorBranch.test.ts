import { expect, it } from "vitest";
import { activeBranchMessages } from "./turnEditorContext";
import type { ChatDetail, Message } from "./types";

function message(id: string, parent_id: string | null, visible = true): Message {
  return { id, parent_id, chat_id: "branch-chat", role: "assistant", status: "complete", parts: [],
    transcript_visible: visible, created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-01T00:00:00Z" };
}
function branch(messages: Message[], head: string) {
  return activeBranchMessages({ messages, active_head_message_id: head } as ChatDetail).map((item) => item.id);
}

it("walks through a hidden parent to retain the earlier visible conversation", () => {
  expect(branch([message("first", null), message("hidden", "first", false),
    message("last", "hidden")], "last")).toEqual(["first", "last"]);
});

it("does not expose a sibling when the selected branch is entirely hidden", () => {
  expect(branch([message("root", null, false), message("hidden", "root", false),
    message("sibling", "root")], "hidden")).toEqual([]);
});

it("finishes a cyclic branch after visiting each identity once", () => {
  expect(branch([message("first", "last"), message("last", "first")], "last"))
    .toEqual(["first", "last"]);
});
