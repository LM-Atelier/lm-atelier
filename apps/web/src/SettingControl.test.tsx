import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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
