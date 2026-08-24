import type { Dispatch, SetStateAction } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";

import type { ComposerDraft, ComposerPromptSource } from "./composerPromptSource";
import { recoverPromptSourceSend } from "./promptSourceSendRecovery";

const source: ComposerPromptSource = {
  version: 1,
  batch_id: "prompt-batch",
  expected_plan_version: 3,
  expected_plan_sha256: "a".repeat(64),
  item_id: "prompt-item",
  expected_review_version: 2,
  expected_reviewed_sha256: "b".repeat(64),
  prompt_template_id: "ptdef-one",
  prompt_template_revision_id: "ptrev-two",
  contract_sha256: "c".repeat(64),
};

function recovery(
  initial: Record<string, ComposerDraft>,
): {
  current: () => Record<string, ComposerDraft>;
  invalidateQueries: ReturnType<typeof vi.fn>;
} {
  let drafts = initial;
  const invalidateQueries = vi.fn().mockResolvedValue(undefined);
  const client = { invalidateQueries } as unknown as QueryClient;
  const setDrafts: Dispatch<SetStateAction<Record<string, ComposerDraft>>> = (update) => {
    drafts = typeof update === "function" ? update(drafts) : update;
  };
  recoverPromptSourceSend(client, setDrafts, {
    chatId: "chat",
    text: "the rejected draft",
    promptSource: source,
  });
  return { current: () => drafts, invalidateQueries };
}

describe("Prompt Library send recovery", () => {
  it("restores an optimistically cleared draft and invalidates its batch", () => {
    const result = recovery({ chat: { text: "", promptSource: null } });

    expect(result.current().chat).toEqual({
      text: "the rejected draft",
      promptSource: source,
    });
    expect(result.invalidateQueries).toHaveBeenCalledWith({
      queryKey: ["prompt-batch", "prompt-batch"],
    });
  });

  it("does not clobber text typed while the rejected send was in flight", () => {
    const typed = { chat: { text: "something new I typed", promptSource: null } };
    const result = recovery(typed);

    expect(result.current()).toBe(typed);
    expect(result.current().chat.text).toBe("something new I typed");
    expect(result.current().chat.promptSource).toBeNull();
  });
});
