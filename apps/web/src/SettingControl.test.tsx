import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SettingControl } from "./SettingControl";
import type { VideoLengthControl } from "./settings";
import type { SettingField } from "./types";

const durationField: SettingField & { video_length: VideoLengthControl } = {
  key: "duration_seconds",
  label: "Length (seconds)",
  type: "number",
  default: 49 / 16,
  minimum: 17 / 16,
  maximum: 81 / 16,
  step: 0.01,
  multiple_of: null,
  choices: [],
  scope: "workflow",
  visibility: "basic",
  restart_required: false,
  available: true,
  unavailable_reason: null,
  help: "Choose a length in seconds.",
  video_length: {
    frames_parameter: "frames",
    fps_parameter: "fps",
    fps_numerator: 16,
    fps_denominator: 1,
    frame_alignment: 16,
    frame_offset: 1,
    minimum_frames: 17,
    maximum_frames: 81,
  },
};

afterEach(cleanup);

const stepsField: SettingField = {
  key: "steps",
  label: "Steps",
  type: "integer",
  default: 30,
  minimum: 1,
  maximum: 200,
  step: 1,
  multiple_of: null,
  choices: [],
  scope: "workflow",
  visibility: "basic",
  restart_required: false,
  available: true,
  unavailable_reason: null,
  help: "",
};

const objectField: SettingField = {
  ...stepsField,
  key: "extra",
  label: "Extra",
  type: "object",
  default: {},
  minimum: null,
  maximum: null,
  step: null,
};

// The shape both settings surfaces use: one keyed wrapper per field, whose
// state is lifted, so an edit re-renders the wrapper with the new value.
function KeyedSurface({ field }: { field: SettingField }) {
  const [settings, setSettings] = useState<Record<string, unknown>>({
    [field.key]: field.default,
  });
  return (
    <div key={`${field.scope}:${field.key}`}>
      <SettingControl
        field={field}
        value={settings[field.key] ?? field.default}
        onChange={(value) => setSettings({ ...settings, [field.key]: value })}
      />
    </div>
  );
}

describe("SettingControl numeric editing", () => {
  it("reports nothing while the box is empty, for either numeric type", () => {
    for (const type of ["integer", "number"] as const) {
      const onChange = vi.fn();
      const field = { ...stepsField, type, key: type };
      render(<SettingControl field={field} value={30} onChange={onChange} />);
      const input = screen.getByRole("spinbutton") as HTMLInputElement;

      fireEvent.change(input, { target: { value: "" } });

      // An emptied integer box used to report NaN, which reaches the API as
      // null and is refused. An emptied number box used to report 0, because
      // `Number("")` is 0 - so it could not be emptied at all, it jumped to
      // zero and every settings surface persists that immediately.
      expect(onChange).not.toHaveBeenCalled();
      expect(input.value).toBe("");
      cleanup();
    }
  });

  it("reports the number once the text is one, and keeps the box showing what was typed", () => {
    const onChange = vi.fn();
    render(<SettingControl field={stepsField} value={30} onChange={onChange} />);
    const input = screen.getByRole("spinbutton") as HTMLInputElement;

    fireEvent.change(input, { target: { value: "" } });
    fireEvent.change(input, { target: { value: "4" } });
    fireEvent.change(input, { target: { value: "45" } });

    expect(onChange.mock.calls.map((call) => call[0])).toEqual([4, 45]);
    expect(input.value).toBe("45");
  });

  it("reports nothing for any half-typed number, not only an emptied box", () => {
    // A number input hands the handler its SANITIZED value, so every one of
    // these arrives as "" rather than as the characters typed. That is why the
    // empty case is the whole class: typing a minus sign, or a decimal point
    // before its digits, used to store 0 in a number field and null in an
    // integer one, on the way to a perfectly ordinary value.
    for (const partial of ["-", "3.", "1e", "abc"]) {
      const onChange = vi.fn();
      render(<SettingControl field={{ ...stepsField, key: "cfg", type: "number" }} value={7} onChange={onChange} />);
      const input = screen.getByRole("spinbutton") as HTMLInputElement;

      fireEvent.change(input, { target: { value: partial } });

      expect(onChange, partial).not.toHaveBeenCalled();
      expect(input.value).toBe("");
      cleanup();
    }
  });

  it("reports a negative number once it is one", () => {
    const onChange = vi.fn();
    render(<SettingControl field={{ ...stepsField, key: "cfg", type: "number", minimum: -10 }} value={7} onChange={onChange} />);
    const input = screen.getByRole("spinbutton") as HTMLInputElement;

    fireEvent.change(input, { target: { value: "-" } });
    fireEvent.change(input, { target: { value: "-2" } });

    expect(onChange.mock.calls.map((call) => call[0])).toEqual([-2]);
  });

  it("shows the stored value again once the field is left", () => {
    const onChange = vi.fn();
    render(<SettingControl field={stepsField} value={30} onChange={onChange} />);
    const input = screen.getByRole("spinbutton") as HTMLInputElement;

    fireEvent.change(input, { target: { value: "" } });
    expect(input.value).toBe("");

    fireEvent.blur(input);

    // Nothing was stored while it was empty, so leaving it restores what is.
    expect(onChange).not.toHaveBeenCalled();
    expect(input.value).toBe("30");
  });
});

describe("SettingControl identity across an edit", () => {
  it("keeps the same input element, and its caret, while a number is typed", () => {
    render(<KeyedSurface field={stepsField} />);
    const before = screen.getByRole("spinbutton");
    before.focus();
    expect(document.activeElement).toBe(before);

    fireEvent.change(before, { target: { value: "31" } });

    // Keying the wrapper on the VALUE replaced this element on every keystroke,
    // so typing a three-digit number lost the caret after the first digit.
    expect(screen.getByRole("spinbutton")).toBe(before);
    expect(document.activeElement).toBe(before);
    expect((before as HTMLInputElement).value).toBe("31");
  });

  it("still refreshes the JSON control when the value changes elsewhere", () => {
    const { rerender } = render(
      <SettingControl field={objectField} value={{ a: 1 }} onChange={vi.fn()} />,
    );
    const first = (screen.getByRole("textbox") as HTMLTextAreaElement).value;
    rerender(
      <SettingControl field={objectField} value={{ a: 2 }} onChange={vi.fn()} />,
    );
    const second = (screen.getByRole("textbox") as HTMLTextAreaElement).value;

    // This control is uncontrolled, so React will not update it from `value`.
    // It carries its own value-bearing key for exactly this reason, and that
    // must survive the wrapper's key becoming stable.
    expect(first).not.toBe(second);
    expect(second).toContain('"a": 2');
  });
});

describe("SettingControl video length", () => {
  it("shows requested and aligned delivered duration when they differ", () => {
    render(<SettingControl field={durationField} value={3} onChange={vi.fn()} />);

    expect(screen.getByText("Requested 3 seconds · delivers 3.0625 seconds (49 frames).")).toBeInTheDocument();
  });

  it("keeps seconds as the user-facing setting value", () => {
    const onChange = vi.fn();
    render(<SettingControl field={durationField} value={3} onChange={onChange} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: /Length/ }), {
      target: { value: "4.25" },
    });
    expect(onChange).toHaveBeenCalledWith(4.25);
  });
});
