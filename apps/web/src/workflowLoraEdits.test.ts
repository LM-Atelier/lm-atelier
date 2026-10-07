/** Saved edits to a workflow's own LoRAs, as a settings layer holds them. */

import { expect, it } from "vitest";
import type { WorkflowLoraControlSlot, WorkflowLoraOverrideTargetWitness } from "./types";
import {
  RESET_EDITS,
  WORKFLOW_LORA_EDITS_KEY,
  effectiveFields,
  hasStaleTarget,
  readStoredEdits,
  withFieldEdit,
  withResetEdits,
  withoutRevisionEdits,
} from "./workflowLoraEdits";

const KEY = WORKFLOW_LORA_EDITS_KEY;

function witness(overrides: Partial<WorkflowLoraOverrideTargetWitness> = {}): WorkflowLoraOverrideTargetWitness {
  return {
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
    ...overrides,
  };
}

const SLOT: WorkflowLoraControlSlot = {
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
};

function layer(value: number, target = witness()): Record<string, unknown> {
  return withFieldEdit({}, target, SLOT, "model_strength", value);
}

it("tells an absent value from a malformed one and a readable one", () => {
  expect(readStoredEdits({ steps: 3 })).toEqual({ status: "absent" });
  expect(readStoredEdits({ [KEY]: { version: 1, targets: "no" } })).toEqual({ status: "invalid" });
  expect(readStoredEdits({ [KEY]: { version: 2, targets: [] } })).toEqual({ status: "invalid" });
  expect(readStoredEdits({ [KEY]: { version: 1, targets: [{ overrides: [] }] } })).toEqual({ status: "invalid" });
  expect(readStoredEdits({ [KEY]: RESET_EDITS })).toEqual({ status: "present", edits: RESET_EDITS });
});

it("resolves each layer in the server's order, profile first and chat last", () => {
  const order = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
  const layers = order.map((value) => layer(value));
  for (let top = 0; top < layers.length; top += 1) {
    const upTo = layers.slice(0, top + 1);
    expect(effectiveFields(upTo, witness(), SLOT).model_strength?.value).toBe(order[top]);
  }
  // Swapping any two neighbours changes the answer whenever the higher one is on top.
  for (let index = 0; index < layers.length - 1; index += 1) {
    const swapped = [...layers.slice(0, index), layers[index + 1], layers[index]];
    expect(effectiveFields(swapped, witness(), SLOT).model_strength?.value).toBe(order[index]);
  }
});

it("lets an empty value hide every layer below it but not the ones above", () => {
  const project = layer(0.3);
  const chatPreset = { [KEY]: RESET_EDITS };
  const chat = withFieldEdit({}, witness(), SLOT, "clip_strength", 0.2);

  const effective = effectiveFields([project, chatPreset, chat], witness(), SLOT);

  expect(effective.model_strength).toBeUndefined();
  expect(effective.clip_strength).toEqual({ value: 0.2, fromThisLayer: true });
});

it("ignores edits made for another workflow, revision or setup", () => {
  const layers = [
    layer(0.3, witness({ workflow_definition_id: "workflow-other" })),
    layer(0.4, witness({ workflow_revision_id: "wfrev-garden-2" })),
    layer(0.5, witness({ activation_witness_sha256: "9".repeat(64) })),
  ];

  expect(effectiveFields(layers, witness(), SLOT)).toEqual({});
});

it("writes an edit under the exact setup, replacing an older setup of the same revision", () => {
  const older = layer(0.3, witness({ activation_witness_sha256: "9".repeat(64) }));
  const other = withFieldEdit(older, witness({ workflow_revision_id: "wfrev-other" }), SLOT, "clip_strength", 1);
  expect(hasStaleTarget(readStoredEdits(other), witness())).toBe(true);

  const edited = withFieldEdit(other, witness(), SLOT, "model_strength", 0.9);

  const stored = readStoredEdits(edited);
  expect(stored.status).toBe("present");
  if (stored.status !== "present") return;
  expect(stored.edits.targets.map((target) => target.workflow_revision_id).sort()).toEqual(["wfrev-garden-1", "wfrev-other"]);
  expect(hasStaleTarget(stored, witness())).toBe(false);
  expect(effectiveFields([edited], witness(), SLOT).model_strength?.value).toBe(0.9);
  expect(readStoredEdits(other).status).toBe("present");
});

it("clears this revision's edits so lower layers show again, and drops the key when nothing is left", () => {
  const project = layer(0.3);
  const chat = { ...layer(0.8), steps: 7 };

  const cleared = withoutRevisionEdits(chat, witness());

  expect(cleared).toEqual({ steps: 7 });
  expect(effectiveFields([project, cleared], witness(), SLOT).model_strength?.value).toBe(0.3);
  const mixed = withFieldEdit(chat, witness({ workflow_revision_id: "wfrev-other" }), SLOT, "model_strength", 1);
  expect(readStoredEdits(withoutRevisionEdits(mixed, witness())).status).toBe("present");
});

it("keeps a reset when clearing, discards an unreadable value, and resets to the exact empty value", () => {
  expect(withoutRevisionEdits({ [KEY]: RESET_EDITS }, witness())).toEqual({ [KEY]: RESET_EDITS });
  expect(withoutRevisionEdits({ [KEY]: "broken", steps: 2 }, witness())).toEqual({ steps: 2 });
  expect(withResetEdits({ steps: 2 })).toEqual({ steps: 2, [KEY]: { version: 1, targets: [] } });
  expect(withResetEdits(layer(0.4))[KEY]).toEqual({ version: 1, targets: [] });
});
