import { useState } from "react";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { WorkflowChoiceDropdown } from "./WorkflowChoiceDropdown";

afterEach(cleanup);
const changed = vi.fn();
const pages = { error: null, isPending: false, isFetchingNextPage: false, isFetchNextPageError: false,
  hasNextPage: false, fetchNextPage: vi.fn(), refetch: vi.fn() };
function Harness({ saving = false, hasNextPage = false }: { saving?: boolean; hasNextPage?: boolean }) {
  const [search, setSearch] = useState("");
  const [value, setValue] = useState("retained");
  return <><label htmlFor="image-choice">Image workflow</label>
    <WorkflowChoiceDropdown id="image-choice" label="Image workflow" browseLabel="image workflows" value={value}
      browse={{ search, setSearch, pages: { ...pages, hasNextPage } }} saving={saving} onChange={next => { changed(next); setValue(next); }}
      options={[{ value: "retained", label: "Existing exact workflow", disabled: true },
        { value: "default", label: "Default" }, { value: "automatic", label: "Auto" },
        { value: "portrait", label: "Portrait finish" }]} />
    <button type="button">Other action</button></>;
}

it("searches in the picker and commits only an explicit keyboard choice", () => {
  changed.mockClear();
  render(<Harness />);
  const picker = screen.getByRole("combobox", { name: "Image workflow" });
  expect(picker).toHaveValue("Existing exact workflow");
  expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  picker.focus();
  fireEvent.click(picker);
  fireEvent.change(picker, { target: { value: "Portrait" } });
  expect(screen.getAllByRole("option")).toHaveLength(1);
  expect(changed).not.toHaveBeenCalled();
  fireEvent.keyDown(picker, { key: "ArrowDown" });
  expect(picker).toHaveAttribute("aria-activedescendant", screen.getByRole("option").id);
  fireEvent.keyDown(picker, { key: "Enter" });
  expect(changed).toHaveBeenCalledExactlyOnceWith("portrait");
  expect(picker).toHaveValue("Portrait finish");
  expect(picker).toHaveFocus();
  expect(picker).toHaveAttribute("aria-expanded", "false");
});

it("dismisses unfinished searches with Escape or focus outside without changing the selection", () => {
  changed.mockClear();
  render(<Harness />);
  const picker = screen.getByRole("combobox");
  picker.focus();
  fireEvent.click(picker);
  fireEvent.change(picker, { target: { value: "No match" } });
  expect(screen.getByRole("status")).toHaveTextContent("No matching workflows.");
  fireEvent.keyDown(picker, { key: "Escape" });
  expect(picker).toHaveValue("Existing exact workflow");
  expect(picker).toHaveFocus();
  fireEvent.click(picker);
  fireEvent.change(picker, { target: { value: "Portrait" } });
  act(() => screen.getByRole("button", { name: "Other action" }).focus());
  expect(picker).toHaveValue("Existing exact workflow");
  expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  expect(changed).not.toHaveBeenCalled();
});

it("skips disabled compatibility choices and keeps the input focusable during saving", () => {
  changed.mockClear();
  const { rerender } = render(<Harness />);
  const picker = screen.getByRole("combobox");
  picker.focus();
  fireEvent.keyDown(picker, { key: "ArrowDown" });
  fireEvent.keyDown(picker, { key: "Enter" });
  expect(changed).toHaveBeenCalledExactlyOnceWith("default");
  rerender(<Harness saving />);
  expect(picker).toHaveFocus();
  expect(picker).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(picker);
  fireEvent.change(picker, { target: { value: "Auto" } });
  fireEvent.keyDown(picker, { key: "ArrowDown" });
  expect(picker).toHaveValue("Default");
  expect(changed).toHaveBeenCalledTimes(1);
});

it("keeps pagination reachable and dismisses it with Escape", () => {
  render(<Harness hasNextPage />);
  const picker = screen.getByRole("combobox");
  fireEvent.click(picker);
  fireEvent.keyDown(picker, { key: "Tab" });
  const more = screen.getByRole("button", { name: "Load more image workflows" });
  act(() => more.focus());
  expect(picker).toHaveAttribute("aria-expanded", "true");
  fireEvent.keyDown(more, { key: "Escape" });
  expect(picker).toHaveFocus();
  expect(picker).toHaveAttribute("aria-expanded", "false");
});
