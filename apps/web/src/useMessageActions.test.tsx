import type { PropsWithChildren } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import { api } from "./api";
import { useMessageActions } from "./useMessageActions";

vi.mock("./api", () => ({
  api: {
    chatItemRemovalImpact: vi.fn(),
    removeChatItemContent: vi.fn(),
    deleteExchange: vi.fn(),
    forkThread: vi.fn(),
  },
}));

it("executes selective removal from a fresh impact revision", async () => {
  vi.mocked(api.chatItemRemovalImpact).mockResolvedValue({
    chat_id: "chat_constructed",
    message_id: "msg_constructed",
    message_revision_id: "1".repeat(64),
    role: "user",
    already_removed: false,
    has_replies: true,
    source_backs_regeneration: true,
    detached_message_part_count: 1,
    detached_response_revision_part_count: 0,
    detached_reference_count: 0,
    detached_references: [],
    detached_references_truncated: false,
    released_artifact_count: 0,
    released_artifact_ids: [],
    released_artifacts_truncated: false,
    retained_artifact_count: 0,
    retained_artifact_ids: [],
    retained_artifacts_truncated: false,
    retained_witness_classes: [],
    forensic_erasure: false,
    execute_authorized: false,
  });
  vi.mocked(api.removeChatItemContent).mockResolvedValue({
    operation_key: "operation_constructed",
    chat_id: "chat_constructed",
    message_id: "msg_constructed",
    message_revision_id: "1".repeat(64),
    content_removed_at: "2026-08-25T12:00:00Z",
    replayed: false,
  });
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  const wrapper = ({ children }: PropsWithChildren) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const { result } = renderHook(
    () => useMessageActions(vi.fn(), vi.fn()),
    { wrapper },
  );

  act(() => result.current.removeItem.mutate("msg_constructed"));

  await waitFor(() => expect(api.removeChatItemContent).toHaveBeenCalledWith(
    "msg_constructed",
    "1".repeat(64),
    expect.any(String),
  ));
  expect(api.chatItemRemovalImpact).toHaveBeenCalledWith("msg_constructed");
});
