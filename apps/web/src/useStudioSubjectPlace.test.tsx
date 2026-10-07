/** Replacing a subject: a step that is stopped or refused, and one that finishes after the picture changed. */

import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import type { StudioApplyPlan } from "./studioApplyPlan";
import { readCutoutMask } from "./studioBackground";
import { createMask, type MaskRaster } from "./studioMasks";
import type { ChatDetail, TurnAccepted } from "./types";
import { useStudioSubjectPlace } from "./useStudioSubjectPlace";

vi.mock("./api", () => ({ api: { upload: vi.fn() } }));
vi.mock("./studioBackground", async (original) => ({
  ...(await original<typeof import("./studioBackground")>()),
  readCutoutMask: vi.fn(),
}));
vi.mock("./studioMasks", async (original) => ({
  ...(await original<typeof import("./studioMasks")>()),
  encodeMaskPng: vi.fn(async () => new Blob(["old subject"], { type: "image/png" })),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const PLAN = {
  words: "Remove the subject.",
  blendSelection: true,
  sendsLightMap: false,
  cutout: {
    words: "Cut the subject out.",
    redraw: "subject",
    reference: new File(["neutral"], "new-subject.png", { type: "image/png" }),
  },
} as StudioApplyPlan;

type Hook = Parameters<typeof useStudioSubjectPlace>;

function sessionWith(status: string, artifactId?: string): ChatDetail {
  return {
    id: "chat-studio",
    messages: [{
      id: "msg-find",
      role: "assistant",
      status,
      parts: artifactId ? [{ type: "image", artifact_id: artifactId, metadata_json: {} }] : [],
    }],
  } as unknown as ChatDetail;
}

/** An apply that takes the turn at once, as the server answers it. */
function takesTheTurn() {
  return vi.fn((...args: unknown[]) => {
    const onAccepted = args[5] as ((accepted: TurnAccepted) => void) | undefined;
    onAccepted?.({ assistant_message: { id: "msg-find" } } as unknown as TurnAccepted);
  });
}

function replacing(apply: ReturnType<typeof vi.fn>, onError = vi.fn(), localEdit = vi.fn()) {
  const hook = renderHook(
    ({ sessionId, session }: { sessionId: string; session: ChatDetail }) =>
      useStudioSubjectPlace(sessionId, session, apply as unknown as Hook[2], localEdit as unknown as Hook[3], onError),
    { initialProps: { sessionId: "chat-studio", session: sessionWith("pending") } },
  );
  const onAccepted = vi.fn();
  act(() => hook.result.current.start(PLAN, "art-1", { width: 8, height: 6 }, onAccepted));
  return { ...hook, onError, onAccepted };
}

describe("a replaced subject whose step ends early", () => {
  it("frees the studio without a word or another step when a step was stopped", () => {
    const apply = takesTheTurn();
    const { result, rerender, onError } = replacing(apply);
    expect(result.current.busy).toBe(true);

    rerender({ sessionId: "chat-studio", session: sessionWith("cancelled") });

    expect(result.current.busy).toBe(false);
    expect(onError).not.toHaveBeenCalled();
    expect(apply).toHaveBeenCalledTimes(1);
  });

  it("frees the studio when a step's turn is refused", () => {
    const apply = vi.fn();
    const { result } = replacing(apply);
    expect(result.current.busy).toBe(true);

    act(() => (apply.mock.calls[0][7] as () => void)());

    expect(result.current.busy).toBe(false);
  });

  it("asks nothing of a picture with no second picture to take a subject from", () => {
    const apply = vi.fn();
    const { result } = renderHook(() =>
      useStudioSubjectPlace("chat-studio", null, apply as unknown as Hook[2], vi.fn() as unknown as Hook[3], vi.fn()));

    act(() => result.current.start({ ...PLAN, cutout: { ...PLAN.cutout!, reference: undefined } }, "art-1", { width: 8, height: 6 }, vi.fn()));

    expect(apply).not.toHaveBeenCalled();
    expect(result.current.busy).toBe(false);
  });

  it("starts nothing more once another picture's session is on screen", async () => {
    let found: (mask: MaskRaster) => void = () => {};
    vi.mocked(readCutoutMask).mockReturnValue(new Promise((resolve) => { found = resolve; }));
    const apply = takesTheTurn();
    const { result, rerender, onError } = replacing(apply);
    rerender({ sessionId: "chat-studio", session: sessionWith("complete", "art-found") });
    expect(readCutoutMask).toHaveBeenCalledWith("art-found", 8, 6);

    // Another picture opened while the old subject was being read.
    rerender({ sessionId: "chat-other", session: sessionWith("pending") });
    expect(result.current.busy).toBe(false);
    const subject = createMask(8, 6);
    subject.data[2 * 8 + 3] = 255;
    await act(async () => found(subject));

    expect(api.upload).not.toHaveBeenCalled();
    expect(apply).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
  });
});
