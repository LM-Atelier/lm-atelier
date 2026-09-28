import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { SettingControl } from "./SettingControl";
import { normalizeSettingsForFields, resolveWorkflowSettings } from "./settings";

afterEach(cleanup);

it.each([
  { type: "integer", value: 81, role: "spinbutton" },
  { type: "number", value: 2.5, role: "spinbutton" },
  { type: "boolean", value: true, role: "checkbox" },
  { type: "string", value: "fixed", role: "textbox" },
])("keeps a fixed $type value in its disabled scalar control", ({ type, value, role }) => {
  const [field] = resolveWorkflowSettings([], { properties: {
    fixed_value: { type, title: "Fixed value", const: value },
  } });
  expect(field).toMatchObject({ type, default: value, choices: [value] });
  render(<SettingControl field={field} value={value} onChange={vi.fn()} />);
  const control = screen.getByRole(role, { name: /Fixed value/ });
  expect(control).toBeDisabled();
  if (typeof value === "boolean") expect(control).toBeChecked();
  else expect(control).toHaveValue(value);
  expect(screen.getByText(`Fixed by this workflow at ${String(value)}.`)).toBeInTheDocument();
});

it.each([
  { type: "integer", choices: [10, 20, 30], saved: 20, selected: 30 },
  { type: "number", choices: [1, 2.5, 7.5], saved: 2.5, selected: 7.5 },
])("offers native $type choices and preserves their numeric values", ({ type, choices, saved, selected }) => {
  const fields = resolveWorkflowSettings([], { properties: {
    native: { type, title: "Native choice", default: saved, enum: choices },
  } });
  expect(fields[0]).toMatchObject({ type: "enum", choices, default: saved });
  const onChange = vi.fn();
  render(<SettingControl field={fields[0]} value={saved} onChange={onChange} />);
  const select = screen.getByRole("combobox", { name: "Native choice" });
  const option = within(select).getByRole("option", { name: String(selected) }) as HTMLOptionElement;
  fireEvent.change(select, { target: { value: option.value } });
  expect(onChange).toHaveBeenCalledWith(selected);
  expect(normalizeSettingsForFields({ native: onChange.mock.calls[0][0] }, fields))
    .toEqual({ native: selected });
});

it("keeps choices with identical labels distinct by their original types", () => {
  const [field] = resolveWorkflowSettings([], { properties: {
    native: { type: "enum", title: "Native choice", default: 1, enum: [1, "1", true] },
  } });
  const onChange = vi.fn();
  render(<SettingControl field={field} value={1} onChange={onChange} />);
  const select = screen.getByRole("combobox", { name: "Native choice" });
  const options = within(select).getAllByRole("option") as HTMLOptionElement[];
  expect(options[0].selected).toBe(true);
  expect(options[1].selected).toBe(false);
  for (const option of options) fireEvent.change(select, { target: { value: option.value } });
  expect(onChange.mock.calls.map(([value]) => value)).toEqual([1, "1", true]);
});
