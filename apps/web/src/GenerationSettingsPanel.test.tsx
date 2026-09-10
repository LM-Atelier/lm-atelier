import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import type { EngineCapabilities, EngineRole } from "./types";

vi.mock("./api", () => ({
  api: {
    workflowRevisionOutputGeometry: vi.fn(),
    resolveWorkflowRevisionOutputGeometry: vi.fn(),
  },
}));

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

// The shape control lives in its own file and is tested there. What only a
// mounted panel can settle is that the panel RENDERS it, hands it the width and
// height the hierarchy resolved, and stores what it answers - so deleting the
// element from the panel fails here rather than passing quietly.
const DIMENSIONED: EngineCapabilities[] = [
  {
    engine: "comfyui",
    roles: ["image"],
    settings: [
      {
        key: "width",
        label: "Width",
        type: "integer",
        default: 1024,
        minimum: 128,
        maximum: 2048,
        step: 64,
        choices: [],
        scope: "workflow",
        visibility: "basic",
        available: true,
      },
      {
        key: "height",
        label: "Height",
        type: "integer",
        default: 768,
        minimum: 128,
        maximum: 2048,
        step: 64,
        choices: [],
        scope: "workflow",
        visibility: "basic",
        available: true,
      },
    ],
    healthy: true,
  } as unknown as EngineCapabilities,
];

function DimensionedPanel({ revisionId }: { revisionId?: string | null }) {
  const [values, setValues] = useState<Record<string, unknown>>({});
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={client}>
      <GenerationSettingsPanel
        role={ROLE}
        engines={DIMENSIONED}
        values={values}
        onValues={setValues}
        presets={[]}
        presetId={null}
        onPreset={vi.fn()}
        workflowRevisionId={revisionId ?? null}
        resetLabel="Reset"
        onReset={vi.fn()}
      />
    </QueryClientProvider>
  );
}

const CAPABILITY = {
  version: 1 as const,
  available: true,
  reason: null,
  revision_id: "rev-1",
  workflow_id: "wf-1",
  artifact_sha256: "a".repeat(64),
  operation: "text_to_image" as const,
  engine: "comfyui" as const,
  size_modes: ["exact" as const, "preset" as const],
  preset_ids: ["16:9" as const, "1:1" as const],
  width: {
    key: "width" as const,
    node_id: "latent",
    input_name: "width" as const,
    default: 1024,
    minimum: 128,
    maximum: 2048,
    multiple_of: 64,
  },
  height: {
    key: "height" as const,
    node_id: "latent",
    input_name: "height" as const,
    default: 768,
    minimum: 128,
    maximum: 2048,
    multiple_of: 64,
  },
  graph_binding_verified: true,
  request_authorized: false as const,
};

describe("the shape control in the panel", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("stores the pixels the server resolved for the ratio that was chosen", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(CAPABILITY);
    vi.mocked(api.resolveWorkflowRevisionOutputGeometry).mockResolvedValue({
      version: 1,
      workflow_id: "wf-1",
      revision_id: "rev-1",
      artifact_sha256: "a".repeat(64),
      operation: "text_to_image",
      engine: "comfyui",
      mode: "image",
      size_mode: "preset",
      preset_id: "16:9",
      width: 1024,
      height: 576,
      graph_binding_verified: true,
      request_authorized: false,
    });

    render(<DimensionedPanel revisionId="rev-1" />);

    // The workflow's own defaults before anything is chosen.
    await waitFor(() => expect(screen.getByText("Output: 1024 × 768")).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "16:9 Wide" }));

    // The height box is the proof: it holds a number nothing in this test typed.
    await waitFor(() => expect(screen.getByLabelText(/Height/i)).toHaveValue(576));
    expect(screen.getByLabelText(/Width/i)).toHaveValue(1024);
    expect(screen.getByText("Output: 1024 × 576")).toBeTruthy();
  });

  it("asks about no revision when the turn pins none", () => {
    render(<DimensionedPanel revisionId={null} />);

    expect(screen.queryByRole("group", { name: "Output aspect ratio" })).toBeNull();
    expect(api.workflowRevisionOutputGeometry).not.toHaveBeenCalled();
  });
});
