import { afterEach, describe, expect, it, vi } from "vitest";
import { cutoutOutcome, readCutoutMask, subjectMask, subjectReach } from "./studioBackground";
import { readSourcePixels } from "./studioSourcePixels";
import type { ChatDetail, Message } from "./types";

vi.mock("./studioSourcePixels", () => ({ readSourcePixels: vi.fn() }));

function session(message?: Partial<Message>): ChatDetail {
  return {
    id: "chat-studio",
    messages: message ? [{ id: "msg-cutout", role: "assistant", parts: [], ...message }] : [],
  } as unknown as ChatDetail;
}

function image(artifactId: string, preview = false) {
  return { type: "image", artifact_id: artifactId, metadata_json: preview ? { preview: true } : {} };
}

describe("cutoutOutcome", () => {
  it("waits until the cutout turn has finished", () => {
    expect(cutoutOutcome(null, "msg-cutout")).toEqual({ state: "waiting" });
    expect(cutoutOutcome(session(), "msg-cutout")).toEqual({ state: "waiting" });
    expect(cutoutOutcome(session({ status: "pending" }), "msg-cutout")).toEqual({ state: "waiting" });
  });

  it("names the finished cutout and never a preview of it", () => {
    const finished = session({
      status: "complete",
      parts: [image("art-preview", true), image("art-cutout")] as Message["parts"],
    });
    expect(cutoutOutcome(finished, "msg-cutout")).toEqual({ state: "ready", artifactId: "art-cutout" });
  });

  it("calls a turn that ended without a picture a failure", () => {
    const empty = session({ status: "complete", parts: [image("art-preview", true)] as Message["parts"] });
    expect(cutoutOutcome(empty, "msg-cutout")).toEqual({ state: "failed" });
    expect(cutoutOutcome(session({ status: "failed" }), "msg-cutout")).toEqual({ state: "failed" });
    expect(cutoutOutcome(session({ status: "cancelled" }), "msg-cutout")).toEqual({ state: "failed" });
  });
});

describe("subjectMask", () => {
  it("takes the subject's coverage from the alpha, soft edges included", () => {
    const pixels = new Uint8ClampedArray([9, 9, 9, 0, 9, 9, 9, 128, 9, 9, 9, 255]);
    const mask = subjectMask(pixels, 3, 1);
    expect([mask.width, mask.height]).toEqual([3, 1]);
    expect(Array.from(mask.data)).toEqual([0, 128, 255]);
  });
});

describe("subjectReach", () => {
  it("gives a new subject room in proportion to the picture, and a little at least", () => {
    expect(subjectReach(1000, 800)).toBe(32);
    expect(subjectReach(800, 1000)).toBe(32);
    expect(subjectReach(100, 100)).toBe(8);
  });
});

describe("readCutoutMask", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.mocked(readSourcePixels).mockReset();
  });

  function decodesTo(width: number, height: number) {
    const close = vi.fn();
    const decode = vi.fn(async () => ({ width, height, close }) as unknown as ImageBitmap);
    // A plain response handing back this exact body. A real Response can return
    // a Blob of another class than the test page's, so the body is checked by
    // identity, never by class.
    const body = new Blob(["png"]);
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, blob: async () => body }) as Response));
    vi.stubGlobal("createImageBitmap", decode);
    return { body, close, decode };
  }

  it("decodes the cutout at the source picture's size", async () => {
    const { body, close, decode } = decodesTo(2, 1);
    vi.mocked(readSourcePixels).mockReturnValue(new Uint8ClampedArray([0, 0, 0, 255, 0, 0, 0, 0]));

    const mask = await readCutoutMask("art-cutout", 2, 1);

    expect(fetch).toHaveBeenCalledWith("/api/artifacts/art-cutout/content");
    expect(decode).toHaveBeenCalledTimes(1);
    const [decoded, options] = decode.mock.calls[0] as unknown as [Blob, ImageBitmapOptions];
    expect(decoded).toBe(body);
    expect(options).toEqual({ resizeWidth: 2, resizeHeight: 1 });
    expect(Array.from(mask?.data ?? [])).toEqual([255, 0]);
    expect(close).toHaveBeenCalled();
  });

  it("refuses a cutout it cannot read or that comes back another size", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("gone", { status: 404 })));
    expect(await readCutoutMask("art-cutout", 2, 1)).toBeNull();

    const { close } = decodesTo(3, 1);
    vi.mocked(readSourcePixels).mockReturnValue(new Uint8ClampedArray(12));
    expect(await readCutoutMask("art-cutout", 2, 1)).toBeNull();
    expect(close).toHaveBeenCalled();

    decodesTo(2, 1);
    vi.mocked(readSourcePixels).mockReturnValue(null);
    expect(await readCutoutMask("art-cutout", 2, 1)).toBeNull();
  });
});
