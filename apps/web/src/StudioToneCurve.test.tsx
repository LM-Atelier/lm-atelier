/** The tone curve: points added, dragged, nudged and taken away on a graph of levels. */

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioToneCurve } from "./StudioToneCurve";
import { curveLevel, toneCurve } from "./studioAdjustments";
import type { StudioCurvePoint } from "./types";

afterEach(cleanup);

function Curve({ start, busy = false, onChange }: {
  start: StudioCurvePoint[];
  busy?: boolean;
  onChange: (points: StudioCurvePoint[]) => void;
}) {
  const [points, setPoints] = useState(start);
  return (
    <StudioToneCurve points={points} busy={busy} onChange={(next) => {
      onChange(next);
      setPoints(next);
    }} />
  );
}

/** The graph, laid out one pixel to a level so a pointer's place is the level it names. */
function graph(container: HTMLElement): SVGSVGElement {
  const svg = container.querySelector("svg");
  if (!svg) throw new Error("no graph");
  svg.getBoundingClientRect = () => ({
    left: 0, top: 0, right: 255, bottom: 255, width: 255, height: 255, x: 0, y: 0, toJSON: () => ({}),
  }) as DOMRect;
  return svg;
}

/** A pointer event at level `x` read, level `y` made. jsdom's own pointer events carry no place. */
function pointer(target: Element, type: string, x: number, y: number) {
  act(() => {
    target.dispatchEvent(new MouseEvent(type, { bubbles: true, button: 0, clientX: x, clientY: 255 - y }));
  });
}

function lastPoints(onChange: ReturnType<typeof vi.fn>): StudioCurvePoint[] {
  return onChange.mock.lastCall?.[0] as StudioCurvePoint[];
}

describe("the tone curve", () => {
  it("adds a point where the graph is pressed and drags it, kept between its neighbours", () => {
    const onChange = vi.fn();
    const { container } = render(<Curve start={[]} onChange={onChange} />);
    const svg = graph(container);

    pointer(svg, "pointerdown", 64, 48);
    expect(lastPoints(onChange)).toEqual([{ x: 64, y: 48 }]);
    pointer(svg, "pointermove", 70, 60);
    pointer(svg, "pointerup", 70, 60);
    expect(lastPoints(onChange)).toEqual([{ x: 70, y: 60 }]);

    pointer(svg, "pointerdown", 192, 208);
    expect(lastPoints(onChange)).toEqual([{ x: 70, y: 60 }, { x: 192, y: 208 }]);
    // Dragged past its neighbour, a point stops just beside it.
    pointer(svg, "pointermove", 10, 250);
    pointer(svg, "pointerup", 10, 250);
    expect(lastPoints(onChange)).toEqual([{ x: 70, y: 60 }, { x: 71, y: 250 }]);
    // A move after the release drags nothing.
    const calls = onChange.mock.calls.length;
    pointer(svg, "pointermove", 200, 100);
    expect(onChange).toHaveBeenCalledTimes(calls);

    // The line drawn is the curve the picture is made with.
    const curve = toneCurve(lastPoints(onChange));
    if (!curve) throw new Error("a curve with points is a curve");
    const line = container.querySelector(".studio-tone-curve-line")?.getAttribute("d") ?? "";
    expect(line).toContain(`L128 ${(255 - curveLevel(curve, 128)).toFixed(2)}`);
  });

  it("takes the point near a press rather than adding another", () => {
    const onChange = vi.fn();
    const { container } = render(<Curve start={[{ x: 64, y: 48 }]} onChange={onChange} />);
    const svg = graph(container);

    pointer(svg, "pointerdown", 70, 52);
    expect(onChange).not.toHaveBeenCalled();
    pointer(svg, "pointermove", 100, 80);
    expect(lastPoints(onChange)).toEqual([{ x: 100, y: 80 }]);
  });

  it("moves a focused point with the arrow keys, and Delete or a double-click takes it away", () => {
    const onChange = vi.fn();
    render(<Curve start={[{ x: 64, y: 48 }, { x: 192, y: 208 }]} onChange={onChange} />);
    const first = screen.getByRole("slider", { name: "Curve point 1" });
    expect(first).toHaveAttribute("aria-valuetext", "64 becomes 48");

    fireEvent.keyDown(first, { key: "ArrowRight" });
    expect(lastPoints(onChange)[0]).toEqual({ x: 65, y: 48 });
    fireEvent.keyDown(screen.getByRole("slider", { name: "Curve point 1" }), { key: "ArrowUp", shiftKey: true });
    expect(lastPoints(onChange)[0]).toEqual({ x: 65, y: 58 });
    // Black and white stay at the ends, so no point reaches either.
    for (let press = 0; press < 7; press += 1) {
      fireEvent.keyDown(screen.getByRole("slider", { name: "Curve point 1" }), { key: "ArrowLeft", shiftKey: true });
    }
    expect(lastPoints(onChange)[0]).toEqual({ x: 1, y: 58 });

    fireEvent.keyDown(screen.getByRole("slider", { name: "Curve point 1" }), { key: "Delete" });
    expect(lastPoints(onChange)).toEqual([{ x: 192, y: 208 }]);
    fireEvent.doubleClick(screen.getByRole("slider", { name: "Curve point 1" }));
    expect(lastPoints(onChange)).toEqual([]);
  });

  it("takes six points at most, and straightens at once", () => {
    const onChange = vi.fn();
    const { container } = render(<Curve start={[]} onChange={onChange} />);
    const svg = graph(container);
    expect(screen.getByRole("button", { name: "Straighten the curve" })).toHaveAttribute("aria-disabled", "true");

    for (const x of [20, 60, 100, 140, 180, 220]) {
      pointer(svg, "pointerdown", x, x + 20);
      pointer(svg, "pointerup", x, x + 20);
    }
    expect(lastPoints(onChange)).toHaveLength(6);
    const calls = onChange.mock.calls.length;
    pointer(svg, "pointerdown", 240, 10);
    expect(onChange).toHaveBeenCalledTimes(calls);
    expect(screen.getByText("A curve takes 6 points; take one away to add another.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Straighten the curve" }));
    expect(lastPoints(onChange)).toEqual([]);
    expect(screen.queryAllByRole("slider")).toHaveLength(0);
  });

  it("draws the picture's brightness behind the curve where it can be read", () => {
    const histogram = new Uint32Array(256);
    histogram[0] = 4;
    histogram[255] = 1;
    const { container, rerender } = render(<StudioToneCurve points={[]} busy={false} onChange={vi.fn()} histogram={histogram} />);

    // The commonest level reaches the top; one a quarter as common, half way.
    const shape = container.querySelector(".studio-tone-curve-histogram")?.getAttribute("d") ?? "";
    expect(shape.startsWith("M0 255 L0 0.00 L1 255.00")).toBe(true);
    expect(shape).toContain("L255 127.50 L255 255 Z");

    rerender(<StudioToneCurve points={[]} busy={false} onChange={vi.fn()} />);
    expect(container.querySelector(".studio-tone-curve-histogram")).toBeNull();
  });

  it("changes nothing while another edit is arriving", () => {
    const onChange = vi.fn();
    const { container } = render(<Curve start={[{ x: 64, y: 48 }]} busy onChange={onChange} />);
    const svg = graph(container);

    pointer(svg, "pointerdown", 150, 150);
    pointer(svg, "pointerdown", 64, 48);
    pointer(svg, "pointermove", 90, 90);
    fireEvent.keyDown(screen.getByRole("slider", { name: "Curve point 1" }), { key: "ArrowUp" });
    fireEvent.keyDown(screen.getByRole("slider", { name: "Curve point 1" }), { key: "Delete" });
    fireEvent.doubleClick(screen.getByRole("slider", { name: "Curve point 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Straighten the curve" }));

    expect(onChange).not.toHaveBeenCalled();
  });
});
