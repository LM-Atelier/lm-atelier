/** Replacing a background: a cutout that is stopped, or that fails. */

import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { StudioApplyPlan } from "./studioApplyPlan";
import type { ChatDetail, TurnAccepted } from "./types";
import { CUTOUT_FAILED, useStudioBackground } from "./useStudioBackground";

afterEach(() => {
  cleanup();
});

const PLAN = {
  words: "Put it on a beach.",
  blendSelection: false,
  sendsLightMap: false,
  cutout: { words: "Cut the subject out.", redraw: "surroundings" },
} as StudioApplyPlan;

function sessionWith(status: string): ChatDetail {
  return {
    id: "chat-studio",
    messages: [{ id: "msg-cutout", role: "assistant", status, parts: [] }],
  } as unknown as ChatDetail;
}

/** An apply that takes the cutout turn at once, as the server answers it. */
function takesTheCutout() {
  return vi.fn((...args: unknown[]) => {
    const onAccepted = args[5] as ((accepted: TurnAccepted) => void) | undefined;
    onAccepted?.({ assistant_message: { id: "msg-cutout" } } as unknown as TurnAccepted);
  });
}

function replacing(apply: ReturnType<typeof takesTheCutout>, onError: (message: string) => void) {
  const hook = renderHook(
    ({ session }: { session: ChatDetail }) =>
      useStudioBackground("chat-studio", session, apply as unknown as Parameters<typeof useStudioBackground>[2], onError),
    { initialProps: { session: sessionWith("pending") } },
  );
  act(() => hook.result.current.start(PLAN, "art-1", { width: 8, height: 6 }, vi.fn()));
  return hook;
}

describe("a replacement whose cutout ends early", () => {
  it("frees the studio without a word or a redraw when the cutout was stopped", () => {
    const apply = takesTheCutout();
    const onError = vi.fn();
    const { result, rerender } = replacing(apply, onError);
    expect(result.current.busy).toBe(true);

    rerender({ session: sessionWith("cancelled") });

    expect(result.current.busy).toBe(false);
    expect(onError).not.toHaveBeenCalled();
    // The cutout only; nothing is redrawn after a stop.
    expect(apply).toHaveBeenCalledTimes(1);
  });

  it("still says so when the cutout failed", () => {
    const apply = takesTheCutout();
    const onError = vi.fn();
    const { result, rerender } = replacing(apply, onError);

    rerender({ session: sessionWith("failed") });

    expect(result.current.busy).toBe(false);
    expect(onError).toHaveBeenCalledWith(CUTOUT_FAILED);
    expect(apply).toHaveBeenCalledTimes(1);
  });
});
