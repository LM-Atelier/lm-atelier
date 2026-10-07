import type { WorkflowLoraControlSlot, WorkflowLoraOverrideTargetWitness } from "./types";

/** The settings key that carries a workflow's own LoRA edits beside its other settings. */
export const WORKFLOW_LORA_EDITS_KEY = "workflow_lora_overrides";

export type WorkflowLoraField = "enabled" | "model_strength" | "clip_strength";

type Changes = Partial<Record<WorkflowLoraField, boolean | number>>;

interface SlotEdit {
  slot_id: string;
  loader_contract: string;
  loader_authority_sha256: string;
  changes: Changes;
}

type TargetEdits = WorkflowLoraOverrideTargetWitness & { overrides: SlotEdit[] };

export interface WorkflowLoraEdits {
  version: 1;
  targets: TargetEdits[];
}

export type StoredEdits =
  | { status: "absent" }
  | { status: "invalid" }
  | { status: "present"; edits: WorkflowLoraEdits };

/** An explicit empty value: the workflow's own values, whatever a lower layer saved. */
export const RESET_EDITS: WorkflowLoraEdits = { version: 1, targets: [] };

const WITNESS_KEYS: Array<keyof WorkflowLoraOverrideTargetWitness> = [
  "workflow_family_id",
  "workflow_definition_id",
  "workflow_variant_key",
  "workflow_revision_id",
  "slot_contract_version",
  "revision_scope_sha256",
  "api_graph_sha256",
  "dependency_contract_sha256",
  "activation_binding_sha256",
  "activation_witness_sha256",
];

const FIELDS: WorkflowLoraField[] = ["enabled", "model_strength", "clip_strength"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Read one settings layer's saved edits without trusting their shape. */
export function readStoredEdits(settings: Record<string, unknown> | null | undefined): StoredEdits {
  if (!settings || !Object.prototype.hasOwnProperty.call(settings, WORKFLOW_LORA_EDITS_KEY)) {
    return { status: "absent" };
  }
  const value = settings[WORKFLOW_LORA_EDITS_KEY];
  if (!isRecord(value) || value.version !== 1 || !Array.isArray(value.targets)) return { status: "invalid" };
  const targets = value.targets;
  const wellFormed = targets.every((target) => isRecord(target)
    && WITNESS_KEYS.every((key) => Object.prototype.hasOwnProperty.call(target, key))
    && Array.isArray(target.overrides)
    && target.overrides.every((override) => isRecord(override)
      && typeof override.slot_id === "string"
      && typeof override.loader_contract === "string"
      && typeof override.loader_authority_sha256 === "string"
      && isRecord(override.changes)));
  return wellFormed ? { status: "present", edits: value as unknown as WorkflowLoraEdits } : { status: "invalid" };
}

export function sameWitness(
  left: WorkflowLoraOverrideTargetWitness,
  right: WorkflowLoraOverrideTargetWitness,
): boolean {
  return WITNESS_KEYS.every((key) => left[key] === right[key]);
}

function namesSameRevision(
  target: WorkflowLoraOverrideTargetWitness,
  witness: WorkflowLoraOverrideTargetWitness,
): boolean {
  return target.workflow_definition_id === witness.workflow_definition_id
    && target.workflow_revision_id === witness.workflow_revision_id;
}

/** Saved edits that name this revision but an older setup of it, which the server refuses. */
export function hasStaleTarget(edits: StoredEdits, witness: WorkflowLoraOverrideTargetWitness): boolean {
  return edits.status === "present"
    && edits.edits.targets.some((target) => namesSameRevision(target, witness) && !sameWitness(target, witness));
}

export interface EffectiveField {
  value: boolean | number;
  fromThisLayer: boolean;
}

/** The value each editable field resolves to for one slot, lowest layer first, as the server orders them.
 *
 * A layer's empty value hides every layer below it. The last layer is the one
 * being edited, so a field it sets is reported as coming from it.
 */
export function effectiveFields(
  layers: ReadonlyArray<Record<string, unknown> | null | undefined>,
  witness: WorkflowLoraOverrideTargetWitness,
  slot: WorkflowLoraControlSlot,
): Partial<Record<WorkflowLoraField, EffectiveField>> {
  const result: Partial<Record<WorkflowLoraField, EffectiveField>> = {};
  layers.forEach((layer, index) => {
    const stored = readStoredEdits(layer);
    if (stored.status !== "present") return;
    if (stored.edits.targets.length === 0) {
      for (const field of FIELDS) delete result[field];
      return;
    }
    const target = stored.edits.targets.find((candidate) => sameWitness(candidate, witness));
    const override = target?.overrides.find((candidate) => candidate.slot_id === slot.slot_id);
    if (!override) return;
    for (const field of FIELDS) {
      const value = override.changes[field];
      if (value !== undefined) result[field] = { value, fromThisLayer: index === layers.length - 1 };
    }
  });
  return result;
}

function copyEdits(stored: StoredEdits): WorkflowLoraEdits {
  if (stored.status !== "present") return { version: 1, targets: [] };
  return JSON.parse(JSON.stringify(stored.edits)) as WorkflowLoraEdits;
}

/** Set one field for one slot in the layer being edited, under this exact witness. */
export function withFieldEdit(
  settings: Record<string, unknown>,
  witness: WorkflowLoraOverrideTargetWitness,
  slot: WorkflowLoraControlSlot,
  field: WorkflowLoraField,
  value: boolean | number,
): Record<string, unknown> {
  const edits = copyEdits(readStoredEdits(settings));
  // An older setup of this same revision is refused by the server, so an edit
  // made against the current setup replaces it rather than sitting beside it.
  edits.targets = edits.targets.filter((target) => !namesSameRevision(target, witness) || sameWitness(target, witness));
  let target = edits.targets.find((candidate) => sameWitness(candidate, witness));
  if (!target) {
    target = { ...pickWitness(witness), overrides: [] };
    edits.targets.push(target);
  }
  let override = target.overrides.find((candidate) => candidate.slot_id === slot.slot_id);
  if (!override) {
    override = {
      slot_id: slot.slot_id,
      loader_contract: slot.loader_contract ?? "",
      loader_authority_sha256: slot.loader_authority_sha256 ?? "",
      changes: {},
    };
    target.overrides.push(override);
  }
  override.changes[field] = value;
  return { ...settings, [WORKFLOW_LORA_EDITS_KEY]: edits };
}

function pickWitness(witness: WorkflowLoraOverrideTargetWitness): WorkflowLoraOverrideTargetWitness {
  return Object.fromEntries(WITNESS_KEYS.map((key) => [key, witness[key]])) as unknown as WorkflowLoraOverrideTargetWitness;
}

/** Remove this revision's edits from the layer being edited, so lower layers show through again.
 *
 * Edits for other workflows stay. When nothing is left the key goes too, which
 * is different from a reset: a reset keeps an empty value that hides lower layers.
 */
export function withoutRevisionEdits(
  settings: Record<string, unknown>,
  witness: WorkflowLoraOverrideTargetWitness,
): Record<string, unknown> {
  const stored = readStoredEdits(settings);
  if (stored.status === "absent") return settings;
  const next = { ...settings };
  if (stored.status === "invalid") {
    delete next[WORKFLOW_LORA_EDITS_KEY];
    return next;
  }
  if (stored.edits.targets.length === 0) return settings;
  const edits = copyEdits(stored);
  edits.targets = edits.targets.filter((target) => !namesSameRevision(target, witness));
  if (edits.targets.length === 0) delete next[WORKFLOW_LORA_EDITS_KEY];
  else next[WORKFLOW_LORA_EDITS_KEY] = edits;
  return next;
}

/** Store the explicit empty value in the layer being edited. */
export function withResetEdits(settings: Record<string, unknown>): Record<string, unknown> {
  return { ...settings, [WORKFLOW_LORA_EDITS_KEY]: { version: 1, targets: [] } };
}
