import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ImagePurposeControl } from "./ImagePurposeControl";

afterEach(cleanup);

it("offers source and reference choices within the attachment", () => {
  const change = vi.fn();
  render(<ImagePurposeControl value={undefined} onChange={change} />);
  fireEvent.change(screen.getByRole("combobox", { name: "Picture purpose" }), { target: { value: "edit_source" } });
  expect(change).toHaveBeenLastCalledWith("edit_source");
  fireEvent.change(screen.getByRole("combobox"), { target: { value: "reference" } });
  expect(change).toHaveBeenLastCalledWith("reference");
  fireEvent.change(screen.getByRole("combobox"), { target: { value: "automatic" } });
  expect(change).toHaveBeenLastCalledWith(undefined);
});

it("keeps focus and ignores changes while the turn is being accepted", () => {
  const change = vi.fn();
  render(<ImagePurposeControl value="edit_source" disabled onChange={change} />);
  const control = screen.getByRole("combobox");
  control.focus();
  fireEvent.change(control, { target: { value: "reference" } });
  expect(change).not.toHaveBeenCalled();
  expect(document.activeElement).toBe(control);
  expect(control).toHaveAttribute("aria-disabled", "true");
});
