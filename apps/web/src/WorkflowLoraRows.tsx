import type { ReactNode } from "react";
import type { WorkflowLoraControlSlot, WorkflowLoraControls } from "./types";
import { useWorkflowLoraControls } from "./useWorkflowLoraControls";
import {
  WORKFLOW_LORA_EDITS_KEY,
  effectiveFields,
  hasStaleTarget,
  readStoredEdits,
  sameWitness,
  withFieldEdit,
  withResetEdits,
  withoutRevisionEdits,
  type WorkflowLoraField,
} from "./workflowLoraEdits";

/** The settings layers a panel resolves, lowest first, with the one it edits last. */
export interface WorkflowLoraEditing {
  layers: ReadonlyArray<Record<string, unknown> | null | undefined>;
  values: Record<string, unknown>;
  onValues: (values: Record<string, unknown>, changedKeys?: string[]) => void;
  clearLabel: string;
}

const UNCONFIRMED_CORE = "It could not be confirmed as a standard ComfyUI loader.";
const UNCONFIRMED_PACKAGE = "Its node pack could not be confirmed as a version checked for this.";
const UNREADABLE = "Its saved settings could not be read reliably.";

/** Why a LoRA cannot be changed here, in words, by the fixed reason the server gives. */
const REASONS: Record<string, string> = {
  required_workflow_dependency: "The workflow requires it.",
  missing_asset_binding: "Its file is not installed for this workflow.",
  no_safe_edit_fields: "Nothing about this loader can safely be changed.",
  unsupported_loader_contract: "This kind of loader is not supported yet.",
  missing_core_evidence: UNCONFIRMED_CORE,
  stale_core_evidence: UNCONFIRMED_CORE,
  node_not_core_claimed: UNCONFIRMED_CORE,
  core_loader_contract_mismatch: UNCONFIRMED_CORE,
  unsupported_core_contract: UNCONFIRMED_CORE,
  missing_runtime_binding: UNCONFIRMED_CORE,
  stale_runtime_binding: UNCONFIRMED_CORE,
  invalid_runtime_binding: UNCONFIRMED_CORE,
  missing_package_evidence: UNCONFIRMED_PACKAGE,
  stale_package_evidence: UNCONFIRMED_PACKAGE,
  missing_package_binding: UNCONFIRMED_PACKAGE,
  stale_package_binding: UNCONFIRMED_PACKAGE,
  invalid_package_binding: UNCONFIRMED_PACKAGE,
  package_binding_mismatch: UNCONFIRMED_PACKAGE,
  unsupported_package_contract: UNCONFIRMED_PACKAGE,
  conflicting_package_claim: UNCONFIRMED_PACKAGE,
  package_widget_revision: UNCONFIRMED_PACKAGE,
  invalid_loader_layout: UNREADABLE,
  invalid_loader_link: UNREADABLE,
  invalid_loader_reference: UNREADABLE,
  invalid_loader_strength: UNREADABLE,
  package_widget_layout: UNREADABLE,
};

function name(slot: WorkflowLoraControlSlot): string {
  const reference = slot.asset_binding?.runtime_reference ?? slot.observed_runtime_reference;
  return reference?.split("/").filter(Boolean).at(-1) ?? `LoRA ${slot.position + 1}`;
}

function authored(slot: WorkflowLoraControlSlot): string {
  if (slot.default_enabled === false) return "Off";
  const parts = [];
  if (slot.default_model_strength !== null) parts.push(`Model strength ${slot.default_model_strength}`);
  if (slot.default_clip_strength !== null && slot.strength_mode !== "model_only") {
    parts.push(`CLIP strength ${slot.default_clip_strength}`);
  }
  return parts.length ? parts.join(" · ") : "Settings not known";
}

function note(slot: WorkflowLoraControlSlot, editable: boolean): string {
  if (slot.editability === "editable") {
    return editable ? "" : "Applied as the workflow sets it. Changing it here is not available yet.";
  }
  return (slot.read_only_reason && REASONS[slot.read_only_reason]) || "Its evidence does not allow changing it here.";
}

const FIELD_LABELS: Record<WorkflowLoraField, string> = {
  enabled: "On",
  model_strength: "Model strength",
  clip_strength: "CLIP strength",
};

function authoredValue(slot: WorkflowLoraControlSlot, field: WorkflowLoraField): boolean | number | null {
  if (field === "enabled") return slot.default_enabled;
  return field === "model_strength" ? slot.default_model_strength : slot.default_clip_strength;
}

function SlotEditor({
  controls,
  slot,
  editing,
}: {
  controls: WorkflowLoraControls;
  slot: WorkflowLoraControlSlot;
  editing: WorkflowLoraEditing;
}) {
  const witness = controls.override_target!;
  const effective = effectiveFields(editing.layers, witness, slot);
  const set = (field: WorkflowLoraField, value: boolean | number) => editing.onValues(
    withFieldEdit(editing.values, witness, slot, field, value),
    [WORKFLOW_LORA_EDITS_KEY],
  );
  return (
    <span className="workflow-lora-fields">
      {slot.editable_fields.map((field) => {
        const label = `${name(slot)} ${FIELD_LABELS[field].toLowerCase()}`;
        const current = effective[field]?.value ?? authoredValue(slot, field);
        if (field === "enabled") {
          return (
            <label key={field} className="lora-enabled">
              <input
                type="checkbox"
                aria-label={label}
                checked={current !== false}
                onChange={(event) => set(field, event.target.checked)}
              />
              <span>{FIELD_LABELS[field]}</span>
            </label>
          );
        }
        return (
          <label key={field}>
            <span>{FIELD_LABELS[field]}</span>
            <input
              type="number"
              aria-label={label}
              min={controls.strength_bounds.minimum}
              max={controls.strength_bounds.maximum}
              step={0.05}
              value={typeof current === "number" ? current : ""}
              onChange={(event) => {
                const value = event.target.valueAsNumber;
                if (!Number.isFinite(value)) return;
                const { minimum, maximum } = controls.strength_bounds;
                set(field, Math.min(maximum, Math.max(minimum, value)));
              }}
            />
          </label>
        );
      })}
    </span>
  );
}

function EditActions({ controls, editing }: { controls: WorkflowLoraControls; editing: WorkflowLoraEditing }) {
  const witness = controls.override_target!;
  const stored = readStoredEdits(editing.values);
  const current = stored.status === "present"
    && stored.edits.targets.some((target) => sameWitness(target, witness));
  const stale = hasStaleTarget(stored, witness);
  const discard = () => editing.onValues(withoutRevisionEdits(editing.values, witness), [WORKFLOW_LORA_EDITS_KEY]);
  return (
    <div className="generation-settings-actions" role="group" aria-label="Workflow LoRA actions">
      {stored.status === "invalid" && <p className="muted">Saved LoRA changes here could not be read, so none are applied.</p>}
      {stale && <p className="muted">Saved LoRA changes were made for an earlier setup of this workflow and are not applied.</p>}
      {(stored.status === "invalid" || stale) && (
        <button type="button" className="secondary" onClick={discard}>Discard saved LoRA changes</button>
      )}
      {current && !stale && (
        <button type="button" className="secondary" onClick={discard}>{editing.clearLabel}</button>
      )}
      <button
        type="button"
        className="secondary"
        onClick={() => editing.onValues(withResetEdits(editing.values), [WORKFLOW_LORA_EDITS_KEY])}
      >
        Reset workflow LoRAs
      </button>
    </div>
  );
}

/** The LoRAs the selected workflow applies by itself, shown beside the ones added to it.
 *
 * Read-only: each row says what the workflow applies and, where the server
 * could not confirm it, why. Nothing here names a graph node or a folder, only
 * the file name the workflow uses.
 */
export function WorkflowLoraRows({
  controls,
  unavailable,
  editing,
}: {
  controls: WorkflowLoraControls | null;
  unavailable: boolean;
  editing?: WorkflowLoraEditing;
}) {
  if (unavailable) return <p className="muted">The LoRAs this workflow applies by itself could not be read.</p>;
  if (!controls || controls.slots.length === 0) return null;
  // Edits are offered only while the server names the exact setup they would
  // apply to; without that witness a saved change could not be checked.
  const canEdit = Boolean(editing && controls.override_target);
  return (
    <div role="group" aria-label="In this workflow">
      <strong>In this workflow</strong>
      <ul className="settings-list">
        {controls.slots.map((slot) => {
          const editable = canEdit && slot.editability === "editable" && slot.editable_fields.length > 0;
          const explanation = note(slot, editable);
          return (
            <li key={slot.slot_id} className="lora-stack-item">
              <span>
                <strong>{name(slot)}</strong>
                <small>{authored(slot)}</small>
              </span>
              {editable && <SlotEditor controls={controls} slot={slot} editing={editing!} />}
              {explanation && <small>{explanation}</small>}
            </li>
          );
        })}
      </ul>
      {canEdit && <EditActions controls={controls} editing={editing!} />}
    </div>
  );
}

function Section({ children }: { children: ReactNode }) {
  return (
    <section className="settings-section" aria-label="LoRAs">
      <h4>LoRAs</h4>
      <div className="settings-list">{children}</div>
    </section>
  );
}

function RevisionLorasSection({
  revisionId,
  added,
  editing,
}: {
  revisionId: string;
  added: ReactNode;
  editing?: WorkflowLoraEditing;
}) {
  const { controls, unavailable } = useWorkflowLoraControls(revisionId);
  const own = controls?.slots.length ?? 0;
  if (!added && own === 0) return null;
  return (
    <Section>
      {/* Unreadable is only worth saying where the section is shown anyway. */}
      <WorkflowLoraRows controls={controls} unavailable={unavailable && Boolean(added)} editing={editing} />
      {added}
    </Section>
  );
}

/** The LoRAs section: the workflow's own LoRAs above the ones added to it.
 *
 * The workflow's own are asked for only when a revision is selected, so a
 * panel with no workflow fetches nothing. The section appears when either kind
 * has something to show.
 */
export function LorasSection({
  revisionId,
  children,
  editing,
}: {
  revisionId: string | null;
  children?: ReactNode;
  editing?: WorkflowLoraEditing;
}) {
  if (revisionId !== null) return <RevisionLorasSection revisionId={revisionId} added={children} editing={editing} />;
  return children ? <Section>{children}</Section> : null;
}
