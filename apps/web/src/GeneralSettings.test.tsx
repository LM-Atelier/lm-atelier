/** The General page's notification choice, as somebody meets it. */

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { NOTIFY_WHEN_FINISHED_KEY } from "./completionNotifications";
import { GeneralSettings } from "./GeneralSettings";

function browserAnswers(answer: NotificationPermission) {
  vi.stubGlobal(
    "Notification",
    class {
      static permission: NotificationPermission = "default";
      static requestPermission = vi.fn(async () => answer);
    },
  );
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("starts quiet, and turns on once the browser allows it", async () => {
  browserAnswers("granted");
  render(<GeneralSettings />);
  const group = screen.getByRole("group", { name: "When work finishes" });
  expect(within(group).getByRole("button", { name: "Stay quiet" }).getAttribute("aria-pressed")).toBe("true");

  fireEvent.click(within(group).getByRole("button", { name: "Notify me" }));

  await waitFor(() =>
    expect(within(group).getByRole("button", { name: "Notify me" }).getAttribute("aria-pressed")).toBe("true"),
  );
  expect(localStorage.getItem(NOTIFY_WHEN_FINISHED_KEY)).toBe("on");
});

it("stays quiet and says how to allow it when the browser refuses", async () => {
  browserAnswers("denied");
  render(<GeneralSettings />);
  const group = screen.getByRole("group", { name: "When work finishes" });

  fireEvent.click(within(group).getByRole("button", { name: "Notify me" }));

  expect(await screen.findByRole("status")).toHaveTextContent(/blocked notifications/);
  expect(within(group).getByRole("button", { name: "Stay quiet" }).getAttribute("aria-pressed")).toBe("true");
});
