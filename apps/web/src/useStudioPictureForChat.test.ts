import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useStudioPictureForChat, type StudioPictureForChat } from "./useStudioPictureForChat";

const PICTURE: StudioPictureForChat = { artifactId: "sha256:harbour", artifact: null, origin: "edited" };

function setup(initialChat: string | null = null) {
  const openChat = vi.fn();
  let created: ((chatId: string) => void) | null = null;
  const startChat = vi.fn((onCreated: (chatId: string) => void) => {
    created = onCreated;
  });
  const hook = renderHook(
    ({ displayedChatId }: { displayedChatId: string | null }) =>
      useStudioPictureForChat({ displayedChatId, openChat, startChat }),
    { initialProps: { displayedChatId: initialChat } },
  );
  return { hook, openChat, startChat, finishCreating: (chatId: string) => created?.(chatId) };
}

describe("handing a studio picture to a chat", () => {
  it("opens the chat and hands the picture down as a reference once that chat shows", () => {
    const { hook, openChat } = setup("other");

    act(() => hook.result.current.sendToChat(PICTURE, "harbour-chat"));

    expect(openChat).toHaveBeenCalledWith("harbour-chat");
    expect(hook.result.current.handOver).toBeNull();
    hook.rerender({ displayedChatId: "harbour-chat" });
    expect(hook.result.current.handOver).toMatchObject({
      attachment: { id: "sha256:harbour", kind: "image", artifact: null, origin: "edited" },
      mode: null,
    });
  });

  it("stops handing it down once the chat has taken it", () => {
    const { hook } = setup("harbour-chat");
    act(() => hook.result.current.sendToChat(PICTURE, "harbour-chat"));
    expect(hook.result.current.handOver).not.toBeNull();

    act(() => hook.result.current.taken());

    expect(hook.result.current.handOver).toBeNull();
    hook.rerender({ displayedChatId: "other" });
    hook.rerender({ displayedChatId: "harbour-chat" });
    expect(hook.result.current.handOver).toBeNull();
  });

  it("starts a new chat when none is named, and hands the picture to it once it exists", () => {
    const { hook, openChat, startChat, finishCreating } = setup(null);

    act(() => hook.result.current.sendToChat(PICTURE, null));

    expect(startChat).toHaveBeenCalledTimes(1);
    expect(openChat).not.toHaveBeenCalled();
    act(() => finishCreating("new-chat"));
    hook.rerender({ displayedChatId: "new-chat" });
    expect(hook.result.current.handOver?.attachment.id).toBe("sha256:harbour");
  });
});
