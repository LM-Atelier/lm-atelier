import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { OutputShapeSettings } from "./OutputShapeSettings";
import { OUTPUT_SHAPES_KEY } from "./outputShapePreferences";

afterEach(() => {
  cleanup();
  localStorage.clear();
});

function stored() {
  return JSON.parse(localStorage.getItem(OUTPUT_SHAPES_KEY) ?? "{}");
}

describe("the output shapes setting", () => {
  it("leaves a shape out of the composer for one mode only", () => {
    render(<OutputShapeSettings />);
    const pictures = screen.getByRole("group", { name: "Pictures shapes" });

    fireEvent.click(within(pictures).getByRole("checkbox", { name: "1:1 Square" }));

    expect(within(pictures).getByRole("checkbox", { name: "1:1 Square" })).not.toBeChecked();
    expect(within(screen.getByRole("group", { name: "Videos shapes" })).getByRole("checkbox", { name: "1:1 Square" }))
      .toBeChecked();
    expect(stored().image.hidden).toEqual(["1:1"]);
  });

  it("moves a shape earlier, and goes back to the default on Reset", () => {
    render(<OutputShapeSettings />);
    const videos = screen.getByRole("group", { name: "Videos shapes" });
    const reset = within(videos).getByRole("button", { name: "Reset videos" });
    expect(reset).toHaveAttribute("aria-disabled", "true");

    fireEvent.click(within(videos).getByRole("button", { name: "Move 16:9 Wide earlier for videos" }));

    const order = within(videos).getAllByRole("checkbox").map((box) => box.closest("label")?.textContent);
    expect(order.slice(-2)).toEqual(["16:9 Wide", "3:2 Landscape wide"]);
    expect(reset).toHaveAttribute("aria-disabled", "false");

    fireEvent.click(reset);

    expect(within(videos).getAllByRole("checkbox").at(-1)?.closest("label")?.textContent).toBe("16:9 Wide");
    expect(stored().video.order.at(-1)).toBe("16:9");
  });

  it("does not move the first shape earlier or the last one later", () => {
    render(<OutputShapeSettings />);
    const pictures = screen.getByRole("group", { name: "Pictures shapes" });

    expect(within(pictures).getByRole("button", { name: "Move 1:1 Square earlier for pictures" }))
      .toHaveAttribute("aria-disabled", "true");
    expect(within(pictures).getByRole("button", { name: "Move 16:9 Wide later for pictures" }))
      .toHaveAttribute("aria-disabled", "true");
  });
});
