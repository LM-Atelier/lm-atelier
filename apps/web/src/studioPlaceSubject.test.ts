import { afterEach, describe, expect, it, vi } from "vitest";
import { createMask } from "./studioMasks";
import { encodeRgbaPng, placedSubject, placementIn, readCutoutPixels, solidBox, type RgbaPixels } from "./studioPlaceSubject";
import { readSourcePixels } from "./studioSourcePixels";

vi.mock("./studioSourcePixels", () => ({ readSourcePixels: vi.fn() }));

/** A cutout of the given size, transparent except for the pixels named. */
function cutout(width: number, height: number, pixels: Array<[number, number, [number, number, number, number]]>): RgbaPixels {
  const data = new Uint8ClampedArray(width * height * 4);
  for (const [x, y, rgba] of pixels) data.set(rgba, (y * width + x) * 4);
  return { width, height, data };
}

function pixelAt(data: Uint8ClampedArray, width: number, x: number, y: number): number[] {
  return Array.from(data.slice((y * width + x) * 4, (y * width + x) * 4 + 4));
}

describe("solidBox", () => {
  it("boxes the solid part of a subject and leaves its faint halo out", () => {
    const mask = createMask(6, 4);
    mask.data[1 * 6 + 2] = 255;
    mask.data[2 * 6 + 3] = 128;
    // Faint: a halo the cutout leaves around the subject, not the subject.
    mask.data[0 * 6 + 0] = 127;
    mask.data[3 * 6 + 5] = 40;

    expect(solidBox(mask)).toEqual({ left: 2, top: 1, width: 2, height: 2 });
  });

  it("finds nothing in a cutout that holds no solid subject", () => {
    const mask = createMask(4, 4);
    mask.data[5] = 127;
    expect(solidBox(mask)).toBeNull();
    expect(solidBox(createMask(4, 4))).toBeNull();
  });
});

describe("placementIn", () => {
  it("stands a wide subject on the bottom of a tall place, as wide as the place", () => {
    expect(placementIn({ left: 10, top: 20, width: 40, height: 100 }, 80, 40)).toEqual({
      left: 10, top: 100, width: 40, height: 20,
    });
  });

  it("centres a tall subject across a wide place, as tall as the place", () => {
    expect(placementIn({ left: 0, top: 0, width: 100, height: 50 }, 20, 50)).toEqual({
      left: 40, top: 0, width: 20, height: 50,
    });
  });
});

describe("placedSubject", () => {
  it("draws the subject shrunk into the old one's place and nothing anywhere else", () => {
    // A red square four pixels wide, with an empty margin around it.
    const square = cutout(8, 8, Array.from({ length: 16 }, (_, index) => [
      2 + (index % 4), 2 + Math.floor(index / 4), [200, 0, 0, 255],
    ] as [number, number, [number, number, number, number]]));

    const placed = placedSubject(square, 16, 8, { left: 10, top: 2, width: 2, height: 2 });

    expect(placed).not.toBeNull();
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 16; x += 1) {
        const inside = x >= 10 && x < 12 && y >= 2 && y < 4;
        expect(pixelAt(placed!, 16, x, y)).toEqual(inside ? [200, 0, 0, 255] : [0, 0, 0, 0]);
      }
    }
  });

  it("keeps a soft edge its own colour rather than darkening it with the transparency beside it", () => {
    // Two red pixels with a transparent one between them, shrunk to one pixel:
    // two thirds covered, and still red.
    const sparse = cutout(3, 1, [[0, 0, [255, 0, 0, 255]], [2, 0, [255, 0, 0, 255]]]);

    const placed = placedSubject(sparse, 1, 1, { left: 0, top: 0, width: 1, height: 1 });

    expect(Array.from(placed!)).toEqual([255, 0, 0, 170]);
  });

  it("enlarges a subject smoothly from one pixel to the next", () => {
    const pair = cutout(2, 1, [[0, 0, [255, 0, 0, 255]], [1, 0, [0, 0, 255, 255]]]);

    const placed = placedSubject(pair, 4, 2, { left: 0, top: 0, width: 4, height: 2 });

    expect(pixelAt(placed!, 4, 0, 0)).toEqual([255, 0, 0, 255]);
    expect(pixelAt(placed!, 4, 1, 0)).toEqual([191, 0, 64, 255]);
    expect(pixelAt(placed!, 4, 2, 1)).toEqual([64, 0, 191, 255]);
    expect(pixelAt(placed!, 4, 3, 1)).toEqual([0, 0, 255, 255]);
  });

  it("places nothing when the cutout holds no solid subject", () => {
    const faint = cutout(4, 4, [[1, 1, [255, 255, 255, 100]]]);
    expect(placedSubject(faint, 4, 4, { left: 0, top: 0, width: 4, height: 4 })).toBeNull();
  });
});

describe("readCutoutPixels", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.mocked(readSourcePixels).mockReset();
  });

  it("reads a cutout at its own size, whatever size the picture is", async () => {
    const close = vi.fn();
    const decode = vi.fn(async () => ({ width: 3, height: 2, close }) as unknown as ImageBitmap);
    const body = new Blob(["png"]);
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, blob: async () => body }) as Response));
    vi.stubGlobal("createImageBitmap", decode);
    const pixels = new Uint8ClampedArray(24);
    vi.mocked(readSourcePixels).mockReturnValue(pixels);

    const read = await readCutoutPixels("art-cut");

    expect(fetch).toHaveBeenCalledWith("/api/artifacts/art-cut/content");
    // Decoded as it is: nothing asks for another size.
    expect(decode.mock.calls[0]).toHaveLength(1);
    expect(read).toEqual({ width: 3, height: 2, data: pixels });
    expect(close).toHaveBeenCalled();
  });

  it("gives nothing back for a cutout it cannot fetch or read", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("gone", { status: 404 })));
    expect(await readCutoutPixels("art-cut")).toBeNull();

    const close = vi.fn();
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, blob: async () => new Blob(["png"]) }) as Response));
    vi.stubGlobal("createImageBitmap", vi.fn(async () => ({ width: 3, height: 2, close }) as unknown as ImageBitmap));
    vi.mocked(readSourcePixels).mockReturnValue(null);
    expect(await readCutoutPixels("art-cut")).toBeNull();
    expect(close).toHaveBeenCalled();
  });
});

describe("encodeRgbaPng", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("gives nothing back when the browser cannot draw", async () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    expect(await encodeRgbaPng(2, 2, new Uint8ClampedArray(16))).toBeNull();
  });
});
