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
