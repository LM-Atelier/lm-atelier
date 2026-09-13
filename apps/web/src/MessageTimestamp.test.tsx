/** A message's time, written on the clock somebody chose. */

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { CLOCK_KEY, setClockChoice } from "./clockPreference";
import { MessageTimestamp } from "./MessageTimestamp";

function afternoon(daysAgo: number): string {
  const moment = new Date();
  moment.setDate(moment.getDate() - daysAgo);
  moment.setHours(15, 45, 0, 0);
  return moment.toISOString();
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  cleanup();
});

it("writes today's time and an earlier day's time on a chosen 24-hour clock", () => {
  localStorage.setItem(CLOCK_KEY, "24");
  render(
    <>
      <MessageTimestamp at={afternoon(0)} />
      <MessageTimestamp at={afternoon(3)} />
    </>,
  );

  const [today, earlier] = screen.getAllByRole("time");
  expect(today.textContent).toBe("15:45");
  expect(earlier.textContent).toMatch(/15:45/);
  expect(earlier.textContent).not.toMatch(/\b(AM|PM)\b/i);
  expect(today.getAttribute("title")).toMatch(/15:45/);
});

it("repaints a time already on screen when the clock changes", () => {
  localStorage.setItem(CLOCK_KEY, "12");
  render(<MessageTimestamp at={afternoon(0)} />);
  expect(screen.getByRole("time").textContent).not.toBe("15:45");

  act(() => setClockChoice("24"));

  expect(screen.getByRole("time").textContent).toBe("15:45");
});
