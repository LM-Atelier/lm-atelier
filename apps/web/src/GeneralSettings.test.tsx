/** The General page's notification and sound choices, as somebody meets them. */

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { NOTIFY_WHEN_FINISHED_KEY } from "./completionNotifications";
import { CHIME_TONES, SOUND_WHEN_FINISHED_KEY } from "./completionSound";
import { browserWithAudio } from "./completionSound.test-support";
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

it("plays the chime once when it is chosen, and remembers it", () => {
  const audio = browserWithAudio();
  render(<GeneralSettings />);
  const group = screen.getByRole("group", { name: "Sound" });
  expect(within(group).getByRole("button", { name: "Silent" }).getAttribute("aria-pressed")).toBe("true");

  fireEvent.click(within(group).getByRole("button", { name: "Play a chime" }));

  expect(within(group).getByRole("button", { name: "Play a chime" }).getAttribute("aria-pressed")).toBe("true");
  expect(localStorage.getItem(SOUND_WHEN_FINISHED_KEY)).toBe("on");
  expect(audio.tones).toEqual([...CHIME_TONES.completed]);
});

it("says so, and stays silent, in a browser that cannot play sounds", () => {
  vi.stubGlobal("AudioContext", undefined);
  render(<GeneralSettings />);
  const group = screen.getByRole("group", { name: "Sound" });

  fireEvent.click(within(group).getByRole("button", { name: "Play a chime" }));

  expect(screen.getByRole("status")).toHaveTextContent("This browser cannot play sounds.");
  expect(within(group).getByRole("button", { name: "Silent" }).getAttribute("aria-pressed")).toBe("true");
});
