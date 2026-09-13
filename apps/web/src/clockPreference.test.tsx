/** Writing times on the clock somebody chose, and repainting when they choose again. */

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  CLOCK_KEY,
  clockOptions,
  isClockChoice,
  setClockChoice,
  useClockChoice,
  type ClockChoice,
} from "./clockPreference";

const AFTERNOON = new Date(2026, 0, 1, 15, 45);
const JUST_AFTER_MIDNIGHT = new Date(2026, 0, 1, 0, 5);

function time(date: Date, choice: ClockChoice, locale: string): string {
  return new Intl.DateTimeFormat(locale, {
    hour: "numeric",
    minute: "2-digit",
    ...clockOptions(choice),
  }).format(date);
}

function Reader() {
  const clock = useClockChoice();
  return <output>{time(AFTERNOON, clock, "en-US")}</output>;
}

describe("clock preference", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    cleanup();
  });

  it("leaves the clock to the language until somebody chooses", () => {
    render(<Reader />);

    expect(screen.getByRole("status").textContent).toBe(time(AFTERNOON, "system", "en-US"));
    expect(clockOptions("system")).toEqual({});
  });

  it("writes a 24-hour time even where the language uses 12 hours, and midnight as 00", () => {
    expect(time(AFTERNOON, "24", "en-US")).toBe("15:45");
    expect(time(JUST_AFTER_MIDNIGHT, "24", "en-US")).toBe("00:05");
  });

  it("writes a 12-hour time even where the language uses 24 hours", () => {
    expect(time(AFTERNOON, "12", "de-DE")).toMatch(/^3:45\s?PM$/i);
    expect(time(AFTERNOON, "system", "de-DE")).toBe("15:45");
  });

  it("remembers the choice and repaints what is already on screen", () => {
    render(<Reader />);

    act(() => setClockChoice("24"));

    expect(localStorage.getItem(CLOCK_KEY)).toBe("24");
    expect(screen.getByRole("status").textContent).toBe("15:45");
  });

  it("follows a choice made in another window of the workspace", () => {
    render(<Reader />);

    act(() => {
      localStorage.setItem(CLOCK_KEY, "24");
      window.dispatchEvent(new StorageEvent("storage", { key: CLOCK_KEY }));
    });

    expect(screen.getByRole("status").textContent).toBe("15:45");
  });

  it("treats anything but the three choices as not having chosen", () => {
    localStorage.setItem(CLOCK_KEY, "13");
    render(<Reader />);

    expect(screen.getByRole("status").textContent).toBe(time(AFTERNOON, "system", "en-US"));
    expect(["system", "12", "24"].every(isClockChoice)).toBe(true);
    expect(isClockChoice("13")).toBe(false);
    expect(isClockChoice(24)).toBe(false);
  });
});
