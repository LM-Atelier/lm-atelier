/** The LoRAs a selected workflow applies by itself, shown in generation settings. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import type { WorkflowLoraControlSlot, WorkflowLoraControls } from "./types";
import { LorasSection, WorkflowLoraRows } from "./WorkflowLoraRows";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { workflowLoraControls: vi.fn() } };
});

const REVISION = "wfrev_garden_1";

async function scopeOf(revisionId: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(revisionId));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function slot(overrides: Partial<WorkflowLoraControlSlot> = {}): WorkflowLoraControlSlot {
  return {
    slot_id: `wflora_${"a".repeat(64)}`,
    position: 0,
    loader_type: "LoraLoader",
    loader_contract: "comfy-core-lora-loader-v1",
    loader_authority_sha256: "b".repeat(64),
    editability: "editable",
    read_only_reason: null,
    dependency_required: false,
    observed_runtime_reference: "styles/watercolor-wash.safetensors",
    asset_binding: {
      dependency_slot: "style",
      requirement_key: "default",
      resource_identity_sha256: "c".repeat(64),
      runtime_reference: "styles/watercolor-wash.safetensors",
      sha256: "d".repeat(64),
    },
    default_enabled: true,
    default_model_strength: 0.75,
    default_clip_strength: 0.5,
    strength_mode: "separate",
    editable_fields: ["model_strength", "clip_strength"],
    ...overrides,
  };
}

async function controls(slots: WorkflowLoraControlSlot[], revisionId = REVISION): Promise<WorkflowLoraControls> {
  return {
    version: 1,
    override_contract_version: 1,
    strength_bounds: { minimum: -4, maximum: 4 },
    override_target: null,
    revision_scope_sha256: await scopeOf(revisionId),
    api_graph_sha256: "e".repeat(64),
    dependency_contract_sha256: "f".repeat(64),
    activation_binding_sha256: null,
    ordering_authority: "presentation_only",
    evidence_gaps: [],
    slots,
  };
}

function withQueries(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

beforeEach(() => {
  vi.mocked(api.workflowLoraControls).mockReset();
});

afterEach(() => {
  cleanup();
});

it("names each LoRA by its file and shows what the workflow applies", async () => {
  render(<WorkflowLoraRows controls={await controls([slot()])} unavailable={false} />);

  const rows = within(screen.getByRole("group", { name: "In this workflow" })).getAllByRole("listitem");
  expect(rows).toHaveLength(1);
  expect(rows[0]).toHaveTextContent("watercolor-wash.safetensors");
  expect(rows[0]).not.toHaveTextContent("styles/");
  expect(rows[0]).toHaveTextContent("Model strength 0.75 · CLIP strength 0.5");
  expect(rows[0]).toHaveTextContent("Applied as the workflow sets it. Changing it here is not available yet.");
});

it("shows no CLIP strength for a model-only loader and says when a LoRA is authored off", async () => {
  render(
    <WorkflowLoraRows
      controls={await controls([
        slot({ strength_mode: "model_only", default_clip_strength: 1, loader_type: "LoraLoaderModelOnly" }),
        slot({ slot_id: `wflora_${"9".repeat(64)}`, position: 1, default_enabled: false }),
      ])}
      unavailable={false}
    />,
  );

  const [modelOnly, off] = screen.getAllByRole("listitem");
  expect(modelOnly).toHaveTextContent("Model strength 0.75");
  expect(modelOnly).not.toHaveTextContent("CLIP");
  expect(off).toHaveTextContent("Off");
});

it("says why a LoRA cannot be changed, in words, and never shows a raw reason", async () => {
  render(
    <WorkflowLoraRows
      controls={await controls([
        slot({ editability: "required_locked", read_only_reason: "required_workflow_dependency" }),
        slot({ slot_id: `wflora_${"1".repeat(64)}`, position: 1, editability: "detected_read_only", read_only_reason: "node_not_core_claimed" }),
        slot({ slot_id: `wflora_${"2".repeat(64)}`, position: 2, editability: "detected_read_only", read_only_reason: "some_future_reason" }),
        slot({ slot_id: `wflora_${"3".repeat(64)}`, position: 3, editability: "detected_read_only", read_only_reason: "missing_asset_binding", asset_binding: null, observed_runtime_reference: null }),
      ])}
      unavailable={false}
    />,
  );

  const [required, unconfirmed, unknown, missing] = screen.getAllByRole("listitem");
  expect(required).toHaveTextContent("The workflow requires it.");
  expect(unconfirmed).toHaveTextContent("It could not be confirmed as a standard ComfyUI loader.");
  expect(unknown).toHaveTextContent("Its evidence does not allow changing it here.");
  expect(missing).toHaveTextContent("LoRA 4");
  expect(missing).toHaveTextContent("Its file is not installed for this workflow.");
  expect(screen.getByRole("group", { name: "In this workflow" })).not.toHaveTextContent(/_/);
});

it("shows a workflow's own LoRAs even when nothing can be added to it", async () => {
  vi.mocked(api.workflowLoraControls).mockResolvedValue(await controls([slot()]));
  withQueries(<LorasSection revisionId={REVISION} />);

  expect(await screen.findByRole("region", { name: "LoRAs" })).toHaveTextContent("watercolor-wash.safetensors");
  expect(api.workflowLoraControls).toHaveBeenCalledWith(REVISION, expect.anything());
});

it("shows nothing for a workflow without LoRAs that takes none, and asks nothing without a workflow", async () => {
  vi.mocked(api.workflowLoraControls).mockResolvedValue(await controls([]));
  withQueries(<LorasSection revisionId={REVISION} />);
  await waitFor(() => expect(api.workflowLoraControls).toHaveBeenCalled());
  expect(screen.queryByRole("region", { name: "LoRAs" })).toBeNull();
  cleanup();

  withQueries(<LorasSection revisionId={null}><p>Added LoRAs</p></LorasSection>);
  expect(screen.getByRole("region", { name: "LoRAs" })).toHaveTextContent("Added LoRAs");
  expect(api.workflowLoraControls).toHaveBeenCalledTimes(1);
});

it("keeps the added LoRAs and says the workflow's own could not be read", async () => {
  vi.mocked(api.workflowLoraControls).mockRejectedValue(new Error("unprojectable"));
  withQueries(<LorasSection revisionId={REVISION}><p>Added LoRAs</p></LorasSection>);

  expect(await screen.findByText("The LoRAs this workflow applies by itself could not be read.")).toBeInTheDocument();
  expect(screen.getByText("Added LoRAs")).toBeInTheDocument();
});

it("says nothing when the workflow's own LoRAs cannot be read and none can be added", async () => {
  vi.mocked(api.workflowLoraControls).mockRejectedValue(new Error("unprojectable"));
  withQueries(<LorasSection revisionId={REVISION} />);

  await waitFor(() => expect(api.workflowLoraControls).toHaveBeenCalled());
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(screen.queryByRole("region", { name: "LoRAs" })).toBeNull();
  expect(screen.queryByText(/could not be read/)).toBeNull();
});

it("stops showing a workflow's LoRAs when reading them again fails", async () => {
  vi.mocked(api.workflowLoraControls).mockResolvedValueOnce(await controls([slot()]));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><LorasSection revisionId={REVISION}><p>Added LoRAs</p></LorasSection></QueryClientProvider>);
  expect(await screen.findByText("watercolor-wash.safetensors")).toBeInTheDocument();

  vi.mocked(api.workflowLoraControls).mockRejectedValue(new Error("gone"));
  await client.refetchQueries({ queryKey: ["workflows", "lora-controls", REVISION] }).catch(() => undefined);

  expect(await screen.findByText("The LoRAs this workflow applies by itself could not be read.")).toBeInTheDocument();
  expect(screen.queryByText("watercolor-wash.safetensors")).toBeNull();
});

it("does not keep showing old LoRAs of a workflow that takes none once reading them fails", async () => {
  vi.mocked(api.workflowLoraControls).mockResolvedValueOnce(await controls([slot()]));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><LorasSection revisionId={REVISION} /></QueryClientProvider>);
  expect(await screen.findByText("watercolor-wash.safetensors")).toBeInTheDocument();

  vi.mocked(api.workflowLoraControls).mockRejectedValue(new Error("gone"));
  await client.refetchQueries({ queryKey: ["workflows", "lora-controls", REVISION] }).catch(() => undefined);

  await waitFor(() => expect(screen.queryByText("watercolor-wash.safetensors")).toBeNull());
  expect(screen.queryByRole("region", { name: "LoRAs" })).toBeNull();
});

it("does not show an answer that names a different revision", async () => {
  vi.mocked(api.workflowLoraControls).mockResolvedValue(await controls([slot()], "wfrev_some_other"));
  withQueries(<LorasSection revisionId={REVISION}><p>Added LoRAs</p></LorasSection>);

  expect(await screen.findByText("The LoRAs this workflow applies by itself could not be read.")).toBeInTheDocument();
  expect(screen.queryByText("watercolor-wash.safetensors")).toBeNull();
});
