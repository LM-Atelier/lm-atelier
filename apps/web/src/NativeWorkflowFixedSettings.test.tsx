import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import { normalizeSettingsForFields, resolveWorkflowSettings } from "./settings";
import type { EngineCapabilities, SettingField } from "./types";

afterEach(cleanup);
const role = "image" as const;

const fields: SettingField[] = ["seed", "width", "steps"].map((key) => ({
  key, label: key[0].toUpperCase() + key.slice(1), type: "integer", default: 30,
  minimum: 0, maximum: 4096, step: 1, choices: [], scope: "workflow", visibility: "basic",
  restart_required: false, available: true, unavailable_reason: null, help: "",
}));
const engines: EngineCapabilities[] = [{
  engine: "comfyui", version: "1", roles: ["image"], operations: ["text_to_image"],
  formats: [], devices: [], streaming: false, tool_calling: false, healthy: true, settings: fields, details: {},
}];

it("offers only the current control when a preserved declaration has no graph binding", () => {
  const parameter = "workflow_seed_stage";
  const schema = { type: "object", properties: {
    seed: { type: "integer", default: 14 },
    [parameter]: { type: "integer", default: 99, title: "Seed (Source)",
      "x-lm-atelier-visibility": "basic" },
  }, "x-lm-atelier-graph-settings": { version: 1,
    bindings: [{ node_id: "1", input_name: "seed", parameter }], unbound_parameters: ["seed"],
  } };
  const onValues = vi.fn();
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <GenerationSettingsPanel role={role} engines={engines} workflowSchema={schema}
      values={{}} onValues={onValues} presets={[]} presetId={null} onPreset={vi.fn()}
      profileValues={{ seed: 999 }} resetLabel="Reset" onReset={vi.fn()} />
  </QueryClientProvider>);
  expect(screen.queryByRole("spinbutton", { name: "Seed" })).not.toBeInTheDocument();
  const current = screen.getByRole("spinbutton", { name: "Seed (Source)" });
  expect(current).toHaveValue(99);
  fireEvent.change(current, { target: { value: "77" } });
  expect(onValues).toHaveBeenCalledWith({ [parameter]: 77 }, [parameter]);
  expect(normalizeSettingsForFields({ seed: 999, [parameter]: 77 },
    resolveWorkflowSettings(fields, schema))).toEqual({ [parameter]: 77 });
});

it.each([
  ["linked", "Supplied by a connected node in the workflow."],
  ["primitive", "Fixed by a primitive node in the workflow."],
])("explains a %s setting without offering an ineffective input", (reason, explanation) => {
  const schema = { type: "object", properties: { steps: { type: "integer", default: 20 } },
    "x-lm-atelier-graph-settings": { version: 1,
      bindings: [{ node_id: "1", input_name: "steps", parameter: "steps" }],
      fixed: [{ node_id: "1", input_name: "seed", label: "Seed (Sampler)", reason }],
    },
  };
  const onValues = vi.fn();
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <GenerationSettingsPanel role={role} engines={engines} workflowSchema={schema}
      values={{}} onValues={onValues} presets={[]} presetId={null} onPreset={vi.fn()}
      profileValues={{ seed: 999, width: 2048 }} resetLabel="Reset" onReset={vi.fn()} />
  </QueryClientProvider>);
  expect(screen.getByText("Seed (Sampler)")).toBeInTheDocument();
  expect(screen.getByText(explanation)).toBeInTheDocument();
  expect(screen.queryByRole("spinbutton", { name: "Seed" })).not.toBeInTheDocument();
  expect(screen.queryByRole("spinbutton", { name: "Width" })).not.toBeInTheDocument();
  const steps = screen.getByRole("spinbutton", { name: "Steps" });
  expect(steps).toHaveValue(20);
  fireEvent.change(steps, { target: { value: "21" } });
  expect(onValues).toHaveBeenCalledWith({ steps: 21 }, ["steps"]);
  expect(normalizeSettingsForFields({ seed: 999, width: 2048, steps: 21 },
    resolveWorkflowSettings(fields, schema))).toEqual({ steps: 21 });
});
