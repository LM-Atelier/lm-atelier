import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, renderHook, waitFor } from "@testing-library/react";
import { useStudioImage } from "./useStudioImage";

function fakeBitmap(width = 4, height = 4) {
  return { width, height, close: vi.fn() } as unknown as ImageBitmap;
}

beforeEach(() => {
  vi.stubGlobal("createImageBitmap", vi.fn().mockResolvedValue(fakeBitmap()));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("loading the picture on the studio canvas", () => {
  it("reports a refusal instead of waiting forever", async () => {
    // An error response has a body, and it decodes to nothing. Reported as a
    // missing bitmap it was indistinguishable from one still arriving.
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));

    const { result } = renderHook(() => useStudioImage("art-1"));

    await waitFor(() => expect(result.current.error).toContain("404"));
    expect(result.current.bitmap).toBeNull();
  });

  it("closes the picture it replaces rather than leaking it", async () => {
    const first = fakeBitmap(2, 2);
    const second = fakeBitmap(3, 3);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, blob: () => Promise.resolve({}) }));
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(second),
    );

    const { result, rerender } = renderHook(
      ({ id }: { id: string }) => useStudioImage(id),
      { initialProps: { id: "art-1" } },
    );
    await waitFor(() => expect(result.current.bitmap).toBe(first));

    rerender({ id: "art-2" });

    await waitFor(() => expect(result.current.bitmap).toBe(second));
    expect(first.close).toHaveBeenCalled();
  });

  it("never hands back a picture belonging to a different artifact", async () => {
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(new Promise(() => {})));

    const { result, rerender } = renderHook(
      ({ id }: { id: string }) => useStudioImage(id),
      { initialProps: { id: "art-1" } },
    );
    rerender({ id: "art-2" });

    expect(result.current.bitmap).toBeNull();
  });
});
