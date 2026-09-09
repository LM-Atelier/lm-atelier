import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import type { EngineCapabilities, EngineRole } from "./types";

/**
 * A claim about React's reconciliation, which only a mounted tree can settle.
 *
 * Every control in this panel used to carry a key built from its own VALUE, so
 * each edit produced a new key and React threw the element away and built
 * another. The caret went with it.
 */

const ROLE: EngineRole = "image";

const ENGINES: EngineCapabilities[] = [
  {
    engine: "comfyui",
    roles: ["image"],
    settings: [
      {
        key: "steps",
        label: "Steps",
        type: "integer",
        default: 30,
        minimum: 1,
        maximum: 200,
        step: 1,
        choices: [],
        scope: "workflow",
        visibility: "basic",
        available: true,
      },
    ],
    healthy: true,
  } as unknown as EngineCapabilities,
];

function Panel() {
  const [values, setValues] = useState<Record<string, unknown>>({ steps: 30 });
  return (
    <GenerationSettingsPanel
      role={ROLE}
      engines={ENGINES}
      values={values}
      onValues={setValues}
      presets={[]}
      presetId={null}
      onPreset={vi.fn()}
      resetLabel="Reset"
      onReset={vi.fn()}
    />
  );
}

afterEach(cleanup);

// A workflow whose closed vocabulary's default is NOT its first entry, which is
// the ordinary case rather than a contrived one: the choices are the node's
// declared order and the default is the value the workflow was saved with.
// KSampler declares nine schedulers beginning with "simple", with "karras"
// third.
const NARROWED: EngineCapabilities[] = [
  {
    engine: "comfyui",
    roles: ["image"],
    settings: [
      {
        key: "scheduler",
        label: "Scheduler",
        type: "enum",
        default: "karras",
        choices: ["simple", "karras", "normal"],
        scope: "workflow",
        visibility: "basic",
        available: true,
      },
      {
        key: "steps",
        label: "Steps",
        type: "integer",
        default: 8,
        minimum: 1,
        maximum: 10,
        step: 1,
        choices: [],
        scope: "workflow",
        visibility: "basic",
        available: true,
      },
    ],
    healthy: true,
  } as unknown as EngineCapabilities,
];

function NarrowedPanel({ stored }: { stored: Record<string, unknown> }) {
  return (
    <GenerationSettingsPanel
      role={ROLE}
      engines={NARROWED}
      values={{}}
      onValues={vi.fn()}
      presets={[]}
      presetId={null}
      onPreset={vi.fn()}
      profileValues={stored}
      resetLabel="Reset"
      onReset={vi.fn()}
    />
  );
}

describe("GenerationSettingsPanel", () => {
  it("does not replace a field's input while the user is typing in it", () => {
    render(<Panel />);
    const input = screen.getByLabelText(/Steps/i);
    input.focus();
    expect(document.activeElement).toBe(input);

    fireEvent.change(input, { target: { value: "31" } });

    expect(screen.getByLabelText(/Steps/i)).toBe(input);
    expect(document.activeElement).toBe(input);
    expect((input as HTMLInputElement).value).toBe("31");
  });

  it("still shows a stored value the workflow accepts", () => {
    render(<NarrowedPanel stored={{ scheduler: "normal", steps: 4 }} />);

    expect(screen.getByLabelText(/Scheduler/i)).toHaveValue("normal");
    expect(screen.getByLabelText(/Steps/i)).toHaveValue(4);
  });

  it("shows the default, not the first choice, when the vocabulary lost the stored value", () => {
    // The server drops this layer value and falls back to the field's default.
    // A select whose value matches no option falls to its FIRST option, so
    // without the same filter the panel would say "simple" while the run used
    // "karras" - a setting the user did not choose and is not being given.
    render(<NarrowedPanel stored={{ scheduler: "dpmpp_3m_sde" }} />);

    expect(screen.getByLabelText(/Scheduler/i)).toHaveValue("karras");
  });

  it("shows the default, not the stored number, when the workflow narrowed the range", () => {
    // The same disagreement without a select: the panel would show 50 against a
    // node that will not take more than 10, while the run used 8.
    render(<NarrowedPanel stored={{ steps: 50 }} />);

    expect(screen.getByLabelText(/Steps/i)).toHaveValue(8);
  });
});
