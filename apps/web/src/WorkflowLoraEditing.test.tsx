/** Changing a workflow's own LoRAs from generation settings. */

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, expect, it, vi } from "vitest";
import type { WorkflowLoraControlSlot, WorkflowLoraControls, WorkflowLoraOverrideTargetWitness } from "./types";
import { WORKFLOW_LORA_EDITS_KEY, withFieldEdit } from "./workflowLoraEdits";
import { WorkflowLoraRows } from "./WorkflowLoraRows";

const KEY = WORKFLOW_LORA_EDITS_KEY;

afterEach(() => {
  cleanup();
});

const WITNESS: WorkflowLoraOverrideTargetWitness = {
  workflow_family_id: null,
  workflow_definition_id: "workflow-garden",
  workflow_variant_key: null,
  workflow_revision_id: "wfrev-garden-1",
  slot_contract_version: 1,
  revision_scope_sha256: "1".repeat(64),
  api_graph_sha256: "2".repeat(64),
  dependency_contract_sha256: "3".repeat(64),
  activation_binding_sha256: "4".repeat(64),
  activation_witness_sha256: "5".repeat(64),
};

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
    observed_runtime_reference: "styles/watercolor.safetensors",
    asset_binding: null,
    default_enabled: true,
    default_model_strength: 0.75,
    default_clip_strength: 0.5,
    strength_mode: "separate",
    editable_fields: ["model_strength", "clip_strength"],
    ...overrides,
  };
}

function controls(slots: WorkflowLoraControlSlot[], target: WorkflowLoraOverrideTargetWitness | null = WITNESS): WorkflowLoraControls {
  return {
    version: 1,
    override_contract_version: 1,
    strength_bounds: { minimum: -4, maximum: 4 },
    override_target: target,
    revision_scope_sha256: "1".repeat(64),
    api_graph_sha256: "2".repeat(64),
    dependency_contract_sha256: "3".repeat(64),
    activation_binding_sha256: "4".repeat(64),
    ordering_authority: "presentation_only",
    base_model_family: "krea2",
    accepts_added_loras: false,
    evidence_gaps: [],
    slots,
  };
}

function Harness({
  answer,
  lower = {},
  initial = {},
  onSaved,
}: {
  answer: WorkflowLoraControls;
  lower?: Record<string, unknown>;
  initial?: Record<string, unknown>;
  onSaved?: (values: Record<string, unknown>) => void;
}) {
  const [values, setValues] = useState(initial);
  return (
    <WorkflowLoraRows
      controls={answer}
      unavailable={false}
      editing={{
        layers: [lower, values],
        values,
        onValues: (next) => {
          setValues(next);
          onSaved?.(next);
        },
        clearLabel: "Clear chat overrides for current workflow",
      }}
    />
  );
}

it("offers the fields the loader allows, starting from what the workflow sets", () => {
  render(<Harness answer={controls([
    slot(),
    slot({ slot_id: `wflora_${"c".repeat(64)}`, position: 1, loader_type: "LoraLoaderModelOnly", strength_mode: "model_only", default_clip_strength: null, editable_fields: ["model_strength"], observed_runtime_reference: "styles/ink.safetensors" }),
    slot({ slot_id: `wflora_${"d".repeat(64)}`, position: 2, editability: "required_locked", read_only_reason: "required_workflow_dependency", editable_fields: [], observed_runtime_reference: "styles/base.safetensors" }),
  ])} />);

  expect(screen.getByLabelText("watercolor.safetensors model strength")).toHaveValue(0.75);
  expect(screen.getByLabelText("watercolor.safetensors clip strength")).toHaveValue(0.5);
  expect(screen.getByLabelText("ink.safetensors model strength")).toHaveValue(0.75);
  expect(screen.queryByLabelText("ink.safetensors clip strength")).toBeNull();
  expect(screen.queryByLabelText("base.safetensors model strength")).toBeNull();
  expect(screen.getByText("The workflow requires it.")).toBeInTheDocument();
});

it("offers an on and off switch only where the loader has one", () => {
  render(<Harness answer={controls([slot({ loader_type: "Power Lora Loader (rgthree)", loader_contract: "rgthree-power-lora-loader-v1", editable_fields: ["enabled", "model_strength", "clip_strength"] })])} />);

  const toggle = screen.getByLabelText("watercolor.safetensors on");
  expect(toggle).toBeChecked();
  fireEvent.click(toggle);
  expect(screen.getByLabelText("watercolor.safetensors on")).not.toBeChecked();
});

it("saves a strength under the exact workflow setup", () => {
  const onSaved = vi.fn();
  render(<Harness answer={controls([slot()])} onSaved={onSaved} />);

  fireEvent.change(screen.getByLabelText("watercolor.safetensors model strength"), { target: { value: "1.25" } });

  const saved = onSaved.mock.lastCall![0] as Record<string, { targets: Array<Record<string, unknown>> }>;
  const target = saved[KEY].targets[0];
  expect(target.workflow_revision_id).toBe("wfrev-garden-1");
  expect(target.activation_witness_sha256).toBe("5".repeat(64));
  expect(target.overrides).toEqual([{
    slot_id: `wflora_${"a".repeat(64)}`,
    loader_contract: "comfy-core-lora-loader-v1",
    loader_authority_sha256: "b".repeat(64),
    changes: { model_strength: 1.25 },
  }]);
  expect(screen.getByLabelText("watercolor.safetensors model strength")).toHaveValue(1.25);

  fireEvent.change(screen.getByLabelText("watercolor.safetensors clip strength"), { target: { value: "9" } });
  const clamped = onSaved.mock.lastCall![0] as Record<string, { targets: Array<{ overrides: Array<{ changes: Record<string, number> }> }> }>;
  expect(clamped[KEY].targets[0].overrides[0].changes).toEqual({ model_strength: 1.25, clip_strength: 4 });
});

it("clears only this chat's changes so an inherited value shows again", () => {
  const inherited = withFieldEdit({}, WITNESS, slot(), "model_strength", 0.3);
  const chat = withFieldEdit({}, WITNESS, slot(), "model_strength", 0.9);
  const onSaved = vi.fn();
  render(<Harness answer={controls([slot()])} lower={inherited} initial={chat} onSaved={onSaved} />);
  expect(screen.getByLabelText("watercolor.safetensors model strength")).toHaveValue(0.9);

  fireEvent.click(screen.getByRole("button", { name: "Clear chat overrides for current workflow" }));

  expect(onSaved).toHaveBeenLastCalledWith({});
  expect(screen.getByLabelText("watercolor.safetensors model strength")).toHaveValue(0.3);
  expect(screen.queryByRole("button", { name: "Clear chat overrides for current workflow" })).toBeNull();
});

it("resets to the workflow's own values by saving the exact empty value, even over inherited changes", () => {
  const inherited = withFieldEdit({}, WITNESS, slot(), "model_strength", 0.3);
  const onSaved = vi.fn();
  render(<Harness answer={controls([slot()])} lower={inherited} onSaved={onSaved} />);
  expect(screen.getByLabelText("watercolor.safetensors model strength")).toHaveValue(0.3);

  fireEvent.click(screen.getByRole("button", { name: "Reset workflow LoRAs" }));

  expect(onSaved).toHaveBeenLastCalledWith({ [KEY]: { version: 1, targets: [] } });
  expect(screen.getByLabelText("watercolor.safetensors model strength")).toHaveValue(0.75);
});

it("does not carry changes for one workflow into another one", () => {
  const otherWitness = { ...WITNESS, workflow_definition_id: "workflow-ink", workflow_revision_id: "wfrev-ink-1" };
  const saved = withFieldEdit({}, WITNESS, slot(), "model_strength", 1.5);
  render(<Harness answer={controls([slot()], otherWitness)} initial={saved} />);

  expect(screen.getByLabelText("watercolor.safetensors model strength")).toHaveValue(0.75);
  expect(screen.queryByRole("button", { name: "Clear chat overrides for current workflow" })).toBeNull();
});

it("says when saved changes belong to an earlier setup or cannot be read, and discards them", () => {
  const earlier = withFieldEdit({}, { ...WITNESS, activation_witness_sha256: "9".repeat(64) }, slot(), "model_strength", 1.5);
  const onSaved = vi.fn();
  render(<Harness answer={controls([slot()])} initial={earlier} onSaved={onSaved} />);
  const actions = screen.getByRole("group", { name: "Workflow LoRA actions" });
  expect(actions).toHaveTextContent("made for an earlier setup of this workflow and are not applied");
  expect(screen.getByLabelText("watercolor.safetensors model strength")).toHaveValue(0.75);

  fireEvent.click(within(actions).getByRole("button", { name: "Discard saved LoRA changes" }));
  expect(onSaved).toHaveBeenLastCalledWith({});
  cleanup();

  render(<Harness answer={controls([slot()])} initial={{ [KEY]: "unreadable", steps: 4 }} onSaved={onSaved} />);
  expect(screen.getByText("Saved LoRA changes here could not be read, so none are applied.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Discard saved LoRA changes" }));
  expect(onSaved).toHaveBeenLastCalledWith({ steps: 4 });
});

it("offers no changes without an exact setup to name, and says so", () => {
  render(<Harness answer={controls([slot()], null)} />);

  expect(screen.queryByLabelText("watercolor.safetensors model strength")).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset workflow LoRAs" })).toBeNull();
  expect(screen.getByText("Applied as the workflow sets it. Changing it here is not available yet.")).toBeInTheDocument();
});
