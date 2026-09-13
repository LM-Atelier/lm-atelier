import type { ReactNode } from "react";
import type { WorkflowLoraControlSlot, WorkflowLoraControls } from "./types";
import { useWorkflowLoraControls } from "./useWorkflowLoraControls";

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

function note(slot: WorkflowLoraControlSlot): string {
  if (slot.editability === "editable") return "Applied as the workflow sets it. Changing it here is not available yet.";
  return (slot.read_only_reason && REASONS[slot.read_only_reason]) || "Its evidence does not allow changing it here.";
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
}: {
  controls: WorkflowLoraControls | null;
  unavailable: boolean;
}) {
  if (unavailable) return <p className="muted">The LoRAs this workflow applies by itself could not be read.</p>;
  if (!controls || controls.slots.length === 0) return null;
  return (
    <div role="group" aria-label="In this workflow">
      <strong>In this workflow</strong>
      <ul className="settings-list">
        {controls.slots.map((slot) => (
          <li key={slot.slot_id} className="lora-stack-item">
            <span>
              <strong>{name(slot)}</strong>
              <small>{authored(slot)}</small>
            </span>
            <small>{note(slot)}</small>
          </li>
        ))}
      </ul>
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

function RevisionLorasSection({ revisionId, added }: { revisionId: string; added: ReactNode }) {
  const { controls, unavailable } = useWorkflowLoraControls(revisionId);
  const own = controls?.slots.length ?? 0;
  if (!added && own === 0) return null;
  return (
    <Section>
      {/* Unreadable is only worth saying where the section is shown anyway. */}
      <WorkflowLoraRows controls={controls} unavailable={unavailable && Boolean(added)} />
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
export function LorasSection({ revisionId, children }: { revisionId: string | null; children?: ReactNode }) {
  if (revisionId !== null) return <RevisionLorasSection revisionId={revisionId} added={children} />;
  return children ? <Section>{children}</Section> : null;
}
