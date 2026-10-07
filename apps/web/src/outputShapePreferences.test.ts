import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  arrangedShapes,
  DEFAULT_OUTPUT_SHAPES,
  defaultedShape,
  defaultOutputShapes,
  movedShape,
  OUTPUT_SHAPES_KEY,
  setOutputShapeChoice,
  toggledShape,
  useOutputShapes,
} from "./outputShapePreferences";

afterEach(() => {
  cleanup();
  localStorage.clear();
});

describe("the chosen output shapes", () => {
  it("start as every shape, in the server's order, for pictures and videos alike", () => {
    const { result } = renderHook(() => useOutputShapes());

    expect(result.current).toEqual(DEFAULT_OUTPUT_SHAPES);
    expect(result.current.image.order).toEqual(["1:1", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9"]);
  });

  it("fall back to the default for anything stored that is not a whole choice", () => {
    for (const stored of [
      "not json",
      JSON.stringify({ image: { order: ["1:1"], hidden: [] } }),
      JSON.stringify({ image: { order: ["1:1", "1:1", "2:3", "9:16", "4:3", "3:2", "16:9"], hidden: [] } }),
      JSON.stringify({ image: { order: [...DEFAULT_OUTPUT_SHAPES.image.order], hidden: ["5:4"] } }),
    ]) {
      localStorage.setItem(OUTPUT_SHAPES_KEY, stored);
      const { result, unmount } = renderHook(() => useOutputShapes());
      expect(result.current.image).toEqual(DEFAULT_OUTPUT_SHAPES.image);
      unmount();
    }
  });

  it("are remembered per mode and reach whoever is showing them", () => {
    const { result } = renderHook(() => useOutputShapes());
    const wideFirst = movedShape(movedShape(DEFAULT_OUTPUT_SHAPES.video, "16:9", -1), "16:9", -1);

    act(() => setOutputShapeChoice("video", toggledShape(wideFirst, "1:1", false)));

    expect(result.current.video.order.slice(-3)).toEqual(["16:9", "4:3", "3:2"]);
    expect(result.current.video.hidden).toEqual(["1:1"]);
    expect(result.current.image).toEqual(DEFAULT_OUTPUT_SHAPES.image);
    expect(JSON.parse(localStorage.getItem(OUTPUT_SHAPES_KEY) ?? "{}").video.hidden).toEqual(["1:1"]);
  });
});

describe("the default shape", () => {
  it("starts as none, and a stored default is kept only while it is a shape still offered", () => {
    expect(DEFAULT_OUTPUT_SHAPES.image.default).toBeNull();
    const order = [...DEFAULT_OUTPUT_SHAPES.image.order];
    for (const [stored, expected] of [
      [{ order, hidden: [], default: "3:2" }, "3:2"],
      [{ order, hidden: ["3:2"], default: "3:2" }, null],
      [{ order, hidden: [], default: "5:4" }, null],
      [{ order, hidden: [] }, null],
    ] as const) {
      localStorage.setItem(OUTPUT_SHAPES_KEY, JSON.stringify({ image: stored }));
      const { result, unmount } = renderHook(() => useOutputShapes());
      expect(result.current.image.default).toBe(expected);
      unmount();
    }
  });

  it("goes when its shape is left out, and cannot be a shape that is left out", () => {
    const chosen = defaultedShape(DEFAULT_OUTPUT_SHAPES.image, "3:2");
    expect(chosen.default).toBe("3:2");
    expect(toggledShape(chosen, "1:1", false).default).toBe("3:2");
    expect(toggledShape(chosen, "3:2", false).default).toBeNull();
    expect(defaultedShape(toggledShape(DEFAULT_OUTPUT_SHAPES.image, "16:9", false), "16:9").default).toBeNull();
    expect(defaultedShape(chosen, null).default).toBeNull();
  });

  it("is read per mode as a turn is sent", () => {
    expect(defaultOutputShapes()).toEqual({ image: null, video: null });

    setOutputShapeChoice("video", defaultedShape(DEFAULT_OUTPUT_SHAPES.video, "16:9"));

    expect(defaultOutputShapes()).toEqual({ image: null, video: "16:9" });
  });
});

describe("arranging what a workflow offers", () => {
  it("leaves out hidden shapes and follows the chosen order", () => {
    const choice = toggledShape(movedShape(DEFAULT_OUTPUT_SHAPES.image, "4:3", -1), "2:3", false);

    expect(arrangedShapes(["16:9", "4:3", "2:3", "1:1"], choice)).toEqual(["1:1", "4:3", "16:9"]);
  });

  it("offers nothing the workflow cannot make, whatever the choice", () => {
    expect(arrangedShapes(["1:1"], DEFAULT_OUTPUT_SHAPES.image)).toEqual(["1:1"]);
  });

  it("does not move a shape past either end", () => {
    const choice = DEFAULT_OUTPUT_SHAPES.image;

    expect(movedShape(choice, "1:1", -1)).toBe(choice);
    expect(movedShape(choice, "16:9", 1)).toBe(choice);
  });

  it("shows a shape again without listing it twice", () => {
    const hidden = toggledShape(toggledShape(DEFAULT_OUTPUT_SHAPES.image, "1:1", false), "1:1", false);

    expect(hidden.hidden).toEqual(["1:1"]);
    expect(toggledShape(hidden, "1:1", true).hidden).toEqual([]);
  });
});
