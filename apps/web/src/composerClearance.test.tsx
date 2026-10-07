import { cleanup, render } from "@testing-library/react";
import { useRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { COMPOSER_CLEARANCE, useComposerClearance } from "./composerClearance";

function Composer({ active, workflows = false }: { active: boolean; workflows?: boolean }) {
  const wrap = useRef<HTMLDivElement>(null);
  useComposerClearance(wrap, active);
  return (
    <div ref={wrap} data-testid="wrap">
      {workflows && <div className="chat-workflow-choices" data-testid="workflows" />}
      <div className="composer" data-testid="box" />
    </div>
  );
}

/** Where each part sits, by test id; jsdom lays nothing out itself. */
function layout(rects: Record<string, { top: number; bottom: number }>) {
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
    return (rects[this.getAttribute("data-testid") ?? ""] ?? { top: 0, bottom: 0 }) as DOMRect;
  });
}

const published = () => document.documentElement.style.getPropertyValue(COMPOSER_CLEARANCE);

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.documentElement.style.removeProperty(COMPOSER_CLEARANCE);
});

describe("the composer's clearance", () => {
  it("keeps workflow choices clear of jobs as their controls collapse", () => {
    let resized: () => void = () => {};
    vi.stubGlobal("ResizeObserver", class {
      constructor(callback: () => void) { resized = callback; }
      observe() {}
      disconnect() {}
    });
    layout({ wrap: { top: 594, bottom: 800 }, workflows: { top: 594, bottom: 650 }, box: { top: 663, bottom: 782 } });
    render(<Composer active workflows />);
    expect(published()).toBe("206px");

    layout({ wrap: { top: 650, bottom: 800 }, workflows: { top: 650, bottom: 660 }, box: { top: 663, bottom: 782 } });
    resized();
    expect(published()).toBe("150px");
  });

  it("says how far the composer reaches up from the foot of its area", () => {
    layout({ wrap: { top: 594, bottom: 800 }, box: { top: 663, bottom: 782 } });

    render(<Composer active />);

    expect(published()).toBe("137px");
  });

  it("takes the measure back when the composer goes", () => {
    layout({ wrap: { top: 594, bottom: 800 }, box: { top: 663, bottom: 782 } });
    const { unmount } = render(<Composer active />);

    unmount();

    expect(published()).toBe("");
  });

  it("publishes nothing for an editor opened inside the transcript", () => {
    layout({ wrap: { top: 594, bottom: 800 }, box: { top: 663, bottom: 782 } });

    render(<Composer active={false} />);

    expect(published()).toBe("");
  });

  it("measures again as the composer grows", () => {
    let resized: () => void = () => {};
    vi.stubGlobal("ResizeObserver", class {
      constructor(callback: () => void) { resized = callback; }
      observe() {}
      disconnect() {}
    });
    layout({ wrap: { top: 594, bottom: 800 }, box: { top: 663, bottom: 782 } });
    render(<Composer active />);

    // A second line of text: the box starts higher, and the panel must too.
    layout({ wrap: { top: 560, bottom: 800 }, box: { top: 629, bottom: 782 } });
    resized();

    expect(published()).toBe("171px");
  });
});
