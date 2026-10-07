import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CaptionMoveTool,
  captionMiddle,
  DEFAULT_CAPTION,
  draggedShift,
  drawCaption,
} from "./studioCaption";
import type { ImagePoint } from "./studioTools";

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

/** A 2D context that measures every letter as ten pixels wide and lists each step it is asked for. */
function steppingContext() {
  const steps: string[] = [];
  const context = {
    font: "",
    textAlign: "left",
    textBaseline: "alphabetic",
    lineJoin: "miter",
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    shadowColor: "",
    shadowBlur: 0,
    shadowOffsetX: 0,
    shadowOffsetY: 0,
    measureText: (line: string) => ({ width: 10 * line.length }),
    translate: (x: number, y: number) => steps.push(`translate ${x} ${y}`),
    rotate: (angle: number) => steps.push(`rotate ${angle}`),
    strokeText: (line: string, x: number, y: number) => steps.push(`stroke ${line} at ${x} ${y}`),
    fillText: (line: string, x: number, y: number) => steps.push(`fill ${line} at ${x} ${y}`),
  };
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context as never);
  return steps;
}

describe("moving and turning the words", () => {
  it("turns the words about their middle, then moves them as far as they were dragged", async () => {
    const steps = steppingContext();

    await drawCaption(400, 200, {
      ...DEFAULT_CAPTION,
      text: "Harbour\nat dawn",
      anchor: "top_left",
      shift: { x: 0.1, y: -0.05 },
      turn: 90,
    });

    // 16-pixel letters on a 200-pixel picture, 19 from one line to the next,
    // from an 8-pixel margin: both lines are 70 wide, so the block's middle is
    // (43, 27), and the drag is 40 across and 10 up.
    expect(steps).toEqual([
      "translate 83 17",
      `rotate ${Math.PI / 2}`,
      "translate -43 -27",
      "stroke Harbour at 8 8",
      "fill Harbour at 8 8",
      "stroke at dawn at 8 27",
      "fill at dawn at 8 27",
    ]);
  });

  it("draws words neither moved nor turned exactly where their place puts them", async () => {
    const steps = steppingContext();

    await drawCaption(400, 200, { ...DEFAULT_CAPTION, text: "Harbour" });

    expect(steps).toEqual(["stroke Harbour at 200 173", "fill Harbour at 200 173"]);
  });

  it("finds the middle of the block on whichever side of its edge the lines are aligned", () => {
    // Right-aligned at 392: the widest line, 70, reaches back to 322.
    expect(captionMiddle({ x: 392, align: "right", tops: [88] }, [50, 70], 20)).toEqual({ x: 357, y: 100 });
    expect(captionMiddle({ x: 200, align: "center", tops: [10, 34] }, [40], 20)).toEqual({ x: 200, y: 34 });
  });
});

describe("dragging the words", () => {
  function recorded() {
    const said: string[] = [];
    const tool = new CaptionMoveTool({
      hold: () => said.push("hold"),
      drag: (by: ImagePoint) => said.push(`drag ${by.x} ${by.y}`),
      letGo: () => said.push("let go"),
      cancel: () => said.push("cancel"),
    });
    return { said, tool };
  }

  it("takes hold on a press and reports how far the pointer has come since, never where it landed", () => {
    const { said, tool } = recorded();

    tool.down({ x: 300, y: 100 });
    tool.move({ x: 310, y: 95 });
    const changed = tool.up({ x: 320, y: 90 });

    expect(said).toEqual(["hold", "drag 10 -5", "drag 20 -10", "let go"]);
    // Nothing was selected, so there is no stroke for Undo to keep.
    expect(changed).toBe(false);
    expect(tool.cursor).toBe("move");
  });

  it("reports a cancel only for a drag under way, and nothing after it", () => {
    const { said, tool } = recorded();

    tool.cancel();
    tool.move({ x: 50, y: 50 });
    tool.down({ x: 0, y: 0 });
    tool.cancel();
    tool.move({ x: 60, y: 60 });

    expect(tool.up({ x: 60, y: 60 })).toBe(false);
    expect(said).toEqual(["hold", "cancel"]);
  });

  it("moves the words from where the drag took hold, to a whole pixel, and keeps them within a picture either way", () => {
    const size = { width: 400, height: 200 };

    expect(draggedShift({ x: 0.1, y: 0 }, { x: 10.4, y: -5 }, size)).toEqual({ x: 0.125, y: -0.025 });
    expect(draggedShift({ x: 0.5, y: 0.5 }, { x: 4000, y: -4000 }, size)).toEqual({ x: 1, y: -1 });
  });
});
