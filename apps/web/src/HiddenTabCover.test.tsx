/** A tab covered once it is hidden, until someone chooses Show. */

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { HIDDEN_TAB_COVER_KEY, setHiddenTabCover } from "./tabCoverChoice";
import { HiddenTabCover } from "./HiddenTabCover";
import { SensitiveMediaSetting } from "./SensitiveMediaSetting";

let visibility: DocumentVisibilityState = "visible";

function setVisibility(state: DocumentVisibilityState) {
  visibility = state;
  act(() => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
}

function Workspace() {
  const [draft, setDraft] = useState("");
  return (
    <main id="main-content" tabIndex={-1}>
      <label>Draft<input value={draft} onChange={(event) => setDraft(event.target.value)} /></label>
    </main>
  );
}

function show() {
  render(<HiddenTabCover><Workspace /></HiddenTabCover>);
}

beforeEach(() => {
  visibility = "visible";
  vi.spyOn(document, "visibilityState", "get").mockImplementation(() => visibility);
  localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
});

it("never covers the tab while the cover is off", () => {
  show();

  setVisibility("hidden");
  setVisibility("visible");

  expect(screen.queryByRole("heading", { name: "LM Atelier is covered" })).toBeNull();
  expect(screen.getByRole("textbox", { name: "Draft" })).toBeTruthy();
});

it("covers a hidden tab until Show, and keeps what was on it", async () => {
  setHiddenTabCover(true);
  show();
  fireEvent.change(screen.getByRole("textbox", { name: "Draft" }), { target: { value: "half a thought" } });

  setVisibility("hidden");
  setVisibility("visible");

  expect(screen.getByRole("heading", { name: "LM Atelier is covered" })).toBeTruthy();
  // Out of the accessibility tree, not merely painted over.
  expect(screen.queryByRole("textbox", { name: "Draft" })).toBeNull();
  expect(screen.getByRole("button", { name: "Show" })).toHaveFocus();

  fireEvent.click(screen.getByRole("button", { name: "Show" }));

  expect(screen.queryByRole("heading", { name: "LM Atelier is covered" })).toBeNull();
  expect(screen.getByRole("textbox", { name: "Draft" })).toHaveValue("half a thought");
  await vi.waitFor(() => expect(screen.getByRole("main")).toHaveFocus());
});

it("covers a tab that is already hidden when the cover is turned on elsewhere", () => {
  show();
  setVisibility("hidden");

  act(() => {
    setHiddenTabCover(true);
  });
  setVisibility("visible");

  expect(screen.getByRole("heading", { name: "LM Atelier is covered" })).toBeTruthy();
});

it("uncovers the tab when the cover is turned off, and starts afresh when it is turned on again", () => {
  setHiddenTabCover(true);
  show();
  setVisibility("hidden");
  setVisibility("visible");

  act(() => {
    setHiddenTabCover(false);
  });
  expect(screen.getByRole("textbox", { name: "Draft" })).toBeTruthy();

  act(() => {
    setHiddenTabCover(true);
  });
  expect(screen.queryByRole("heading", { name: "LM Atelier is covered" })).toBeNull();
});

it("turns the cover on and off from the privacy settings", () => {
  render(<SensitiveMediaSetting />);
  const toggle = screen.getByRole("checkbox", { name: /Cover this window when it is hidden/ });

  fireEvent.click(toggle);
  expect(localStorage.getItem(HIDDEN_TAB_COVER_KEY)).toBe("on");
  expect(toggle).toBeChecked();

  fireEvent.click(toggle);
  expect(localStorage.getItem(HIDDEN_TAB_COVER_KEY)).toBeNull();
  expect(toggle).not.toBeChecked();
});

it("says so when the browser cannot keep the choice, and changes nothing", () => {
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("storage refused");
  });
  render(<SensitiveMediaSetting />);
  const toggle = screen.getByRole("checkbox", { name: /Cover this window when it is hidden/ });

  fireEvent.click(toggle);

  expect(screen.getByRole("status")).toHaveTextContent("This browser cannot save the choice");
  expect(toggle).not.toBeChecked();
});
