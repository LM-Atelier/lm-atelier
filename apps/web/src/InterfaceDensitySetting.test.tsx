import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AppearanceSettings } from "./AppearanceSettings";
import { DENSITY_KEY } from "./densityPreference";
import { useAppearance } from "./theme";

function Settings() {
  return <AppearanceSettings appearance={useAppearance()} />;
}

beforeEach(() => { localStorage.clear(); });
afterEach(() => { cleanup(); delete document.documentElement.dataset.density; });

describe("interface density", () => {
  it("offers Standard initially and applies a remembered choice to the whole document", () => {
    render(<Settings />);
    const choices = within(screen.getByRole("group", { name: "Interface density" }));
    expect(choices.getByRole("button", { name: "Standard" })).toHaveAttribute("aria-pressed", "true");
    expect(document.documentElement).toHaveAttribute("data-density", "standard");
    choices.getByRole("button", { name: "Compact" }).focus();
    fireEvent.click(choices.getByRole("button", { name: "Compact" }));
    expect(choices.getByRole("button", { name: "Compact" })).toHaveFocus();
    expect(choices.getByRole("button", { name: "Compact" })).toHaveAttribute("aria-pressed", "true");
    expect(document.documentElement).toHaveAttribute("data-density", "compact");
    expect(localStorage.getItem(DENSITY_KEY)).toBe("compact");
    cleanup();
    render(<Settings />);
    expect(within(screen.getByRole("group", { name: "Interface density" }))
      .getByRole("button", { name: "Compact" })).toHaveAttribute("aria-pressed", "true");
  });

  it("applies another window's choice and restores Standard when storage is cleared", () => {
    render(<Settings />);
    localStorage.setItem(DENSITY_KEY, "comfortable");
    fireEvent(window, new StorageEvent("storage", { key: DENSITY_KEY }));
    expect(document.documentElement).toHaveAttribute("data-density", "comfortable");
    localStorage.clear();
    fireEvent(window, new StorageEvent("storage", { key: null }));
    expect(document.documentElement).toHaveAttribute("data-density", "standard");
  });

  it("ignores an unsupported saved value without changing text size", () => {
    localStorage.setItem(DENSITY_KEY, "tiny");
    localStorage.setItem("local-lm-text-size", "larger");
    render(<Settings />);
    expect(document.documentElement).toHaveAttribute("data-density", "standard");
    expect(document.documentElement).toHaveAttribute("data-text-size", "larger");
  });
});
