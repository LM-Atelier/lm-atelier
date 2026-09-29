import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_CAPTION, drawCaption } from "./studioCaption";

afterEach(() => {
  vi.restoreAllMocks();
});

/** A 2D context that remembers each draw with the shadow it was drawn under. */
function recordingContext() {
  const draws: Array<{ call: string; line: string; shadow: string; blur: number; offset: number }> = [];
  const context = {
    font: "",
    textAlign: "left",
    textBaseline: "alphabetic",
    lineJoin: "miter",
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    shadowColor: "rgba(0, 0, 0, 0)",
    shadowBlur: 0,
    shadowOffsetX: 0,
    shadowOffsetY: 0,
    strokeText(line: string) {
      draws.push({ call: "stroke", line, shadow: this.shadowColor, blur: this.shadowBlur, offset: this.shadowOffsetX });
    },
    fillText(line: string) {
      draws.push({ call: "fill", line, shadow: this.shadowColor, blur: this.shadowBlur, offset: this.shadowOffsetX });
    },
  };
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context as never);
  return draws;
}

describe("drawing the words", () => {
  it("casts a shadow once per line, from the outline, and not again from the letters", async () => {
    const draws = recordingContext();

    await drawCaption(400, 200, { ...DEFAULT_CAPTION, text: "Harbour\nat dawn", outline: true, shadow: true });

    // The letters are 16 pixels on a 200-pixel picture at eight percent.
    const cast = { shadow: "rgba(0, 0, 0, 0.55)", blur: 2, offset: 1 };
    const none = { shadow: "rgba(0, 0, 0, 0)", blur: 0, offset: 0 };
    expect(draws).toEqual([
      { call: "stroke", line: "Harbour", ...cast },
      { call: "fill", line: "Harbour", ...none },
      { call: "stroke", line: "at dawn", ...cast },
      { call: "fill", line: "at dawn", ...none },
    ]);
  });

  it("casts it from the letters when there is no outline, and not at all when asked for none", async () => {
    const draws = recordingContext();

    await drawCaption(400, 200, { ...DEFAULT_CAPTION, text: "Harbour", outline: false, shadow: true });
    await drawCaption(400, 200, { ...DEFAULT_CAPTION, text: "Harbour", outline: false, shadow: false });

    expect(draws.map(({ call, shadow }) => [call, shadow])).toEqual([
      ["fill", "rgba(0, 0, 0, 0.55)"],
      ["fill", "rgba(0, 0, 0, 0)"],
    ]);
  });
});
