import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import type { EngineCapabilities } from "./types";

vi.mock("./api", () => ({ api: {
  workflowRevisionOutputGeometry: vi.fn(), workflowLoraControls: vi.fn(),
} }));
afterEach(cleanup);
const role = "image" as const;
const engines: EngineCapabilities[] = [{
  engine: "comfyui", version: "1", roles: [role], operations: ["text_to_image"],
  formats: [], devices: [], streaming: false, tool_calling: false, healthy: true,
  settings: [], details: {},
}];

it.each(["aspect_ratio", "workflow_aspect_ratio_stage"])(
  "offers the native %s choice at basic detail without a conflicting size claim", (parameter) => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockRejectedValue(new Error("No dimension mapping"));
    vi.mocked(api.workflowLoraControls).mockRejectedValue(new Error("No LoRA controls"));
    const schema = { type: "object", properties: {
      [parameter]: { type: "string", title: "Aspect Ratio", default: "4:3",
        enum: ["1:1", "4:3", "3:4"], "x-lm-atelier-visibility": "basic" },
      megapixels: { type: "number", title: "Megapixels", default: 1.25,
        minimum: 0.25, maximum: 4, "x-lm-atelier-visibility": "basic" },
    }, "x-lm-atelier-graph-settings": { version: 1, bindings: [
      { node_id: "1", input_name: "aspect_ratio", parameter },
      { node_id: "1", input_name: "megapixel_budget", parameter: "megapixels" },
    ] } };
    const onValues = vi.fn();
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <GenerationSettingsPanel role={role} engines={engines} workflowSchema={schema}
        workflowRevisionId="native-ratio" values={{}} onValues={onValues}
        presets={[]} presetId={null} onPreset={vi.fn()} resetLabel="Reset" onReset={vi.fn()} />
    </QueryClientProvider>);
    expect(screen.getByRole("button", { name: "basic" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByText("This workflow sets the picture size itself.")).not.toBeInTheDocument();
    const ratio = screen.getByRole("combobox", { name: "Aspect Ratio" });
    expect(ratio).toHaveValue("4:3");
    expect(screen.queryByRole("option", { name: "16:9" })).not.toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Megapixels" })).toHaveValue(1.25);
    fireEvent.change(ratio, { target: { value: "3:4" } });
    expect(onValues).toHaveBeenCalledWith({ [parameter]: "3:4" }, [parameter]);
    expect(api.workflowRevisionOutputGeometry).not.toHaveBeenCalled();
  },
);
