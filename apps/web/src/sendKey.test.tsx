/** Which keystroke sends, and every text field that sends hearing about a change. */

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import {
  SEND_KEY_KEY,
  currentSendKeyChoice,
  isSendKeyChoice,
  isSendKeystroke,
  setSendKeyChoice,
  useSendKeyChoice,
} from "./sendKey";

const key = (overrides: Partial<Pick<KeyboardEvent, "key" | "shiftKey" | "ctrlKey" | "metaKey">> = {}) => ({
  key: "Enter",
  shiftKey: false,
  ctrlKey: false,
  metaKey: false,
  ...overrides,
});

function Reader() {
  return <output>{useSendKeyChoice()}</output>;
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  cleanup();
});

it("sends on Enter by default, as every field always has", () => {
  render(<Reader />);

  expect(screen.getByRole("status").textContent).toBe("enter");
  expect(isSendKeystroke(key(), "enter")).toBe(true);
  expect(isSendKeystroke(key({ ctrlKey: true }), "enter")).toBe(true);
});

it("sends only on Ctrl+Enter or Cmd+Enter when that is chosen, so Enter is free for new lines", () => {
  expect(isSendKeystroke(key(), "mod-enter")).toBe(false);
  expect(isSendKeystroke(key({ ctrlKey: true }), "mod-enter")).toBe(true);
  expect(isSendKeystroke(key({ metaKey: true }), "mod-enter")).toBe(true);
});

it("never sends on Shift+Enter or on any other key", () => {
  for (const choice of ["enter", "mod-enter"] as const) {
    expect(isSendKeystroke(key({ shiftKey: true }), choice)).toBe(false);
    expect(isSendKeystroke(key({ shiftKey: true, ctrlKey: true }), choice)).toBe(false);
    expect(isSendKeystroke(key({ key: "a", ctrlKey: true }), choice)).toBe(false);
  }
});

it("remembers the choice and tells a field that is already open", () => {
  render(<Reader />);

  act(() => setSendKeyChoice("mod-enter"));

  expect(localStorage.getItem(SEND_KEY_KEY)).toBe("mod-enter");
  expect(screen.getByRole("status").textContent).toBe("mod-enter");
});

it("follows a choice made in another window, and ignores values it does not offer", () => {
  render(<Reader />);

  act(() => {
    localStorage.setItem(SEND_KEY_KEY, "mod-enter");
    window.dispatchEvent(new StorageEvent("storage", { key: SEND_KEY_KEY }));
  });
  expect(screen.getByRole("status").textContent).toBe("mod-enter");

  act(() => {
    localStorage.setItem(SEND_KEY_KEY, "space");
    window.dispatchEvent(new StorageEvent("storage", { key: SEND_KEY_KEY }));
  });
  expect(screen.getByRole("status").textContent).toBe("enter");
  expect(isSendKeyChoice("space")).toBe(false);
});

it("never sends while an input method is still composing, under either choice", () => {
  for (const choice of ["enter", "mod-enter"] as const) {
    expect(isSendKeystroke({ ...key({ ctrlKey: true }), nativeEvent: { isComposing: true } }, choice)).toBe(false);
    expect(isSendKeystroke({ ...key({ ctrlKey: true }), keyCode: 229 }, choice)).toBe(false);
  }
});

it("reads the choice as it stands when a keystroke is judged", () => {
  expect(currentSendKeyChoice()).toBe("enter");
  localStorage.setItem(SEND_KEY_KEY, "mod-enter");
  expect(currentSendKeyChoice()).toBe("mod-enter");
});
