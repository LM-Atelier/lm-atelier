/** Selecting the subject: a cutout on the named workflow, whose alpha comes back as a selection. */

import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readCutoutMask } from "./studioBackground";
import { createMask } from "./studioMasks";
import type { ChatDetail, Message, TurnAccepted } from "./types";
import { SUBJECT_NOT_FOUND, SUBJECT_UNREADABLE, useStudioSubjectSelection } from "./useStudioSubjectSelection";

vi.mock("./studioBackground", async (original) => ({
  ...(await original<typeof import("./studioBackground")>()),
  readCutoutMask: vi.fn(),
}));

afterEach(() => {
  cleanup();
});

function sessionWith(status: Message["status"], artifactId?: string): ChatDetail {
  return {
    id: "chat-studio",
    messages: [
      {
        id: "msg-cutout",
        role: "assistant",
        status,
        parts: artifactId ? [{ type: "image", artifact_id: artifactId, metadata_json: {} }] : [],
      },
    ],
  } as unknown as ChatDetail;
}

/** An apply that takes the cutout turn at once, as the server answers it. */
function takesTheCutout() {
  return vi.fn((...args: unknown[]) => {
    const onAccepted = args[5] as ((accepted: TurnAccepted) => void) | undefined;
    onAccepted?.({ assistant_message: { id: "msg-cutout" } } as unknown as TurnAccepted);
  });
}

function finding(apply: ReturnType<typeof vi.fn> = takesTheCutout()) {
  const onError = vi.fn();
  const onFound = vi.fn();
  const hook = renderHook(
    ({ session }: { session: ChatDetail }) =>
      useStudioSubjectSelection(
        "chat-studio",
        session,
        apply as unknown as Parameters<typeof useStudioSubjectSelection>[2],
        onError,
        onFound,
      ),
    { initialProps: { session: sessionWith("pending") } },
  );
  act(() => hook.result.current.start("art-1", { width: 8, height: 6 }, "wfrev_cutout"));
  return { ...hook, apply, onError, onFound };
}

describe("selecting the subject", () => {
  it("cuts the subject out on the named workflow and hands back its alpha at the picture's size, once", async () => {
    const mask = createMask(8, 6);
    vi.mocked(readCutoutMask).mockResolvedValue(mask);
    const { result, rerender, apply, onError, onFound } = finding();

    expect(apply).toHaveBeenCalledTimes(1);
    const [words, source, maskUpload, settings, workflow] = apply.mock.calls[0];
    expect(words).toBe("Cut the subject out onto a transparent background.");
    expect(source).toBe("art-1");
    expect(maskUpload).toBeUndefined();
    expect(settings).toBeUndefined();
    expect(workflow).toBe("wfrev_cutout");
    expect(result.current.busy).toBe(true);

    rerender({ session: sessionWith("complete", "art-cutout") });

    await waitFor(() => expect(onFound).toHaveBeenCalledWith("art-1", mask));
    expect(readCutoutMask).toHaveBeenCalledWith("art-cutout", 8, 6);
    expect(result.current.busy).toBe(false);
    expect(onError).not.toHaveBeenCalled();
    rerender({ session: sessionWith("complete", "art-cutout") });
    expect(readCutoutMask).toHaveBeenCalledTimes(1);
    expect(onFound).toHaveBeenCalledTimes(1);
  });

  it("says so when the cutout failed, and selects nothing", () => {
    const { result, rerender, onError, onFound } = finding();

    rerender({ session: sessionWith("failed") });

    expect(result.current.busy).toBe(false);
    expect(onError).toHaveBeenCalledWith(SUBJECT_NOT_FOUND);
    expect(readCutoutMask).not.toHaveBeenCalled();
    expect(onFound).not.toHaveBeenCalled();
  });

  it("says so when the cutout cannot be read, or its reading fails", async () => {
    vi.mocked(readCutoutMask).mockResolvedValueOnce(null);
    const unread = finding();
    unread.rerender({ session: sessionWith("complete", "art-cutout") });
    await waitFor(() => expect(unread.onError).toHaveBeenCalledWith(SUBJECT_UNREADABLE));
    expect(unread.result.current.busy).toBe(false);
    unread.unmount();

    vi.mocked(readCutoutMask).mockRejectedValueOnce(new Error("decode"));
    const failed = finding();
    failed.rerender({ session: sessionWith("complete", "art-cutout") });
    await waitFor(() => expect(failed.onError).toHaveBeenCalledWith(SUBJECT_UNREADABLE));
    expect(failed.result.current.busy).toBe(false);
    expect(failed.onFound).not.toHaveBeenCalled();
  });

  it("frees the studio without a word when the cutout was stopped", () => {
    const { result, rerender, onError, onFound } = finding();

    rerender({ session: sessionWith("cancelled") });

    expect(result.current.busy).toBe(false);
    expect(onError).not.toHaveBeenCalled();
    expect(onFound).not.toHaveBeenCalled();
  });

  it("asks once while a search is under way, and lets go when the turn is refused", () => {
    const refuses = vi.fn((...args: unknown[]) => (args[7] as () => void)());
    const { result, apply } = finding(refuses);
    expect(result.current.busy).toBe(false);

    const waiting = finding();
    act(() => waiting.result.current.start("art-1", { width: 8, height: 6 }, "wfrev_cutout"));

    expect(apply).toHaveBeenCalledTimes(1);
    expect(waiting.apply).toHaveBeenCalledTimes(1);
    expect(waiting.result.current.busy).toBe(true);
  });
});
