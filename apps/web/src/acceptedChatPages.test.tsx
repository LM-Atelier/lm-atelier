import { QueryClient, QueryClientProvider, type InfiniteData } from "@tanstack/react-query";
import { act, cleanup, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import type { ChatDetail, ChatMessageWindow, ChatTranscriptContext, Message, TurnAccepted } from "./types";
import { useTurnSending } from "./useTurnSending";

afterEach(cleanup);
const stamp = "2026-09-01T00:00:00Z";
function message(id: string, parent: string | null): Message {
  return { id, parent_id: parent, chat_id: "chat-one", role: "assistant", status: "complete",
    parts: [], created_at: stamp, updated_at: stamp };
}
const initial = [message("m0", null), message("m1", "m0"), message("m2", "m1"), message("m3", "m2")];
function setup() {
  const client = new QueryClient();
  client.setQueryData<ChatDetail>(["chat", "chat-one"], {
    id: "chat-one", title: "Conversation", project_id: null, archived: false, pinned: false,
    routing_mode: "auto", confirm_uncertain_media: true, active_chat_profile_id: null,
    active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: "m3",
    created_at: stamp, updated_at: stamp, messages: initial,
  });
  client.setQueryData<InfiniteData<ChatMessageWindow, string | undefined>>(["chat", "chat-one", "messages", "m3"], {
    pages: [{ chat_id: "chat-one", messages: initial.slice(2), has_older: true, has_newer: false },
      { chat_id: "chat-one", messages: initial.slice(0, 2), has_older: false, has_newer: true }],
    pageParams: [undefined, "m2"],
  });
  client.setQueryData<ChatTranscriptContext>(["chat", "chat-one", "context", "m3"], {
    chat_id: "chat-one", head_id: "m3", has_prior_image: true, has_prior_visual: true, has_pending_response: false,
  });
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { result } = renderHook(() => useTurnSending({ client, requestTurnConfirmation: undefined,
    setPendingTurns: vi.fn(), setComposerDrafts: vi.fn() }), { wrapper });
  const apply = (parent: string, activate = true, status: Message["status"] = "pending") => {
    const accepted = { user_message: { ...message("m4", parent), role: "user" },
      assistant_message: { ...message("m5", "m4"), status }, run: {} } as TurnAccepted;
    act(() => result.current.applyAcceptedTurn("chat-one", accepted, activate));
  };
  const pages = (head: string) => client.getQueryData<InfiniteData<ChatMessageWindow>>(["chat", "chat-one", "messages", head]);
  return { client, apply, pages };
}

it("seeds accepted continuation pages before moving the head and retains loaded history", () => {
  const { client, apply, pages } = setup();
  apply("m3");
  expect(pages("m5")?.pages.flatMap((page) => page.messages.map((item) => item.id)))
    .toEqual(["m0", "m1", "m2", "m3", "m4", "m5"]);
  expect(client.getQueryData<ChatDetail>(["chat", "chat-one"])?.active_head_message_id).toBe("m5");
  expect(client.getQueryData<ChatTranscriptContext>(["chat", "chat-one", "context", "m5"]))
    .toMatchObject({ has_prior_image: true, has_pending_response: true });
});

it("does not carry the active sibling's messages into an edited branch", () => {
  const { apply, pages } = setup();
  apply("m0");
  expect(pages("m5")?.pages[0]).toMatchObject({ has_older: true });
  expect(pages("m5")?.pages.flatMap((page) => page.messages.map((item) => item.id))).toEqual(["m4", "m5"]);
  expect(pages("m3")?.pages.flatMap((page) => page.messages.map((item) => item.id))).toEqual(["m2", "m3", "m0", "m1"]);
});

it("keeps the active branch when a saved edit is accepted without activation", () => {
  const { client, apply, pages } = setup();
  apply("m0", false);
  expect(client.getQueryData<ChatDetail>(["chat", "chat-one"])?.active_head_message_id).toBe("m3");
  expect(pages("m5")?.pages[0].messages.map((item) => item.id)).toEqual(["m4", "m5"]);
});

it("replaces a repeated accepted message without appending duplicate identities", () => {
  const { apply, pages } = setup();
  apply("m3");
  apply("m3", true, "complete");
  const messages = pages("m5")?.pages.flatMap((page) => page.messages) ?? [];
  expect(messages.filter((item) => item.id === "m5")).toHaveLength(1);
  expect(messages.find((item) => item.id === "m5")?.status).toBe("complete");
});
