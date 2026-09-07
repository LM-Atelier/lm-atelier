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
});
