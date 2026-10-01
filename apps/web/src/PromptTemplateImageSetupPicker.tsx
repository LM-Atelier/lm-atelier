import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Plus } from "lucide-react";
import { api } from "./api";
import { useReadyWorkflowChoices } from "./useReadyWorkflowChoices";
import { ReadyWorkflowBrowseControls } from "./ReadyWorkflowBrowseControls";
import type { PromptTemplateLora } from "./types";
import type { SimplePromptTemplateResourcePolicy } from "./promptTemplateImageSetup";

const SHA256 = /^[0-9a-f]{64}$/;

type InstalledLora = {
  sha256: string;
  name: string;
  family: string | null;
  modelStrength: number;
  clipStrength: number;
};

function digestFromManifest(manifest: Record<string, unknown>): string | null {
  const digest = manifest.sha256;
  return typeof digest === "string" && SHA256.test(digest) ? digest : null;
}

export function PromptTemplateImageSetupPicker({
  value,
  onChange,
}: {
  value: SimplePromptTemplateResourcePolicy;
  onChange: (value: SimplePromptTemplateResourcePolicy) => void;
}) {
  const workflows = useReadyWorkflowChoices(value.mode === "fixed" ? [value.workflow_revision_id] : []);
  const workflowChoices = workflows.rows.map(row => ({
    id: row.revision_id, label: `${row.family_name} - ${row.workflow_name}`,
  }));
  const assets = useQuery({
    queryKey: ["model-assets", "lora"],
    queryFn: () => api.modelAssets("lora"),
  });
  const installedLoras = useMemo(() => (assets.data ?? []).flatMap((asset): InstalledLora[] => {
    const sha256 = digestFromManifest(asset.manifest_json);
    return asset.active && asset.verified_at && sha256 ? [{
      sha256,
      name: asset.name,
      family: asset.family,
      modelStrength: asset.default_model_strength,
      clipStrength: asset.default_clip_strength,
    }] : [];
  }), [assets.data]);

  const updateStack = (stack: PromptTemplateLora[]) => {
    if (value.mode !== "fixed") return;
    onChange({ ...value, lora_policy: { mode: "fixed", stack } });
  };
  /** One strength of one LoRA, as its box reads. An emptied box, or one holding
   * something that is not a number, is not a strength: the stack keeps the last
   * one, as the workflow LoRA rows do, rather than carrying no strength into a
   * template the server would refuse. */
  const setStrength = (index: number, field: "model_strength" | "clip_strength", entered: number) => {
    if (!Number.isFinite(entered)) return;
    updateStack(value.mode === "fixed" && value.lora_policy.mode === "fixed"
      ? value.lora_policy.stack.map((item, itemIndex) => itemIndex !== index ? item
        : field === "model_strength" ? { ...item, model_strength: entered } : { ...item, clip_strength: entered })
      : []);
  };

  return (
    <section className="prompt-template-image-setup" aria-labelledby="quick-image-setup-heading">
      <h3 id="quick-image-setup-heading">Image setup</h3>
      <label>
        Setup for this template
        <select
          aria-label="Setup for this template"
          value={value.mode}
          onChange={(event) => onChange(event.target.value === "inherited"
            ? { mode: "inherited" }
            : { mode: "fixed", workflow_revision_id: "", lora_policy: { mode: "inherited_auto" } })}
        >
          <option value="inherited">Use this chat's setup (recommended)</option>
          <option value="fixed">Choose a workflow and LoRAs</option>
        </select>
      </label>
      {value.mode === "fixed" && (
        <>
          <ReadyWorkflowBrowseControls workflows={workflows} />
          <label>
            Workflow
            <select
              aria-label="Template workflow"
              value={value.workflow_revision_id}
              onChange={(event) => onChange({ ...value, workflow_revision_id: event.target.value })}
            >
              <option value="">Choose a ready image workflow</option>
              {value.workflow_revision_id && !workflowChoices.some(row => row.id === value.workflow_revision_id)
                && <option value={value.workflow_revision_id}>{workflows.missingLabel}</option>}
              {workflowChoices.map((workflow) => <option key={workflow.id} value={workflow.id}>{workflow.label}</option>)}
            </select>
          </label>
          <label>
            LoRAs
            <select
              aria-label="Template LoRAs"
              value={value.lora_policy.mode}
              onChange={(event) => {
                const mode = event.target.value;
                if (mode === "fixed") {
                  onChange({ ...value, lora_policy: { mode, stack: [] } });
                } else {
                  onChange({ ...value, lora_policy: { mode: mode as "inherited_auto" | "none" } });
                }
              }}
            >
              <option value="inherited_auto">Automatic LoRAs (recommended)</option>
              <option value="none">No LoRAs</option>
              <option value="fixed">Choose installed LoRAs</option>
            </select>
          </label>
          {value.lora_policy.mode === "fixed" && (
            <div className="prompt-lora-editor">
              {value.lora_policy.stack.map((lora, index) => (
                <fieldset key={`${lora.sha256}-${index}`}>
                  <legend>LoRA {index + 1}</legend>
                  <label>
                    Installed LoRA
                    <select
                      aria-label={`Template LoRA ${index + 1}`}
                      value={lora.sha256}
                      onChange={(event) => {
                        const asset = installedLoras.find((candidate) => candidate.sha256 === event.target.value);
                        if (!asset) return;
                        updateStack(value.lora_policy.mode === "fixed" ? value.lora_policy.stack.map((item, itemIndex) => itemIndex === index ? {
                          sha256: asset.sha256,
                          model_strength: asset.modelStrength,
                          clip_strength: asset.clipStrength,
                        } : item) : []);
                      }}
                    >
                      <option value="">Choose an installed LoRA</option>
                      {installedLoras.map((asset) => <option key={asset.sha256} value={asset.sha256}>{asset.name}{asset.family ? ` - ${asset.family}` : ""}</option>)}
                    </select>
                  </label>
                  <details>
                    <summary>Adjust strength</summary>
                    <label>Model strength<input aria-label={`Template LoRA ${index + 1} model strength`} type="number" min={-4} max={4} step="0.05" value={lora.model_strength} onChange={(event) => setStrength(index, "model_strength", event.target.valueAsNumber)} /></label>
                    <label>CLIP strength<input aria-label={`Template LoRA ${index + 1} CLIP strength`} type="number" min={-4} max={4} step="0.05" value={lora.clip_strength} onChange={(event) => setStrength(index, "clip_strength", event.target.valueAsNumber)} /></label>
                  </details>
                  <button type="button" className="secondary compact-button" onClick={() => updateStack(value.lora_policy.mode === "fixed" ? value.lora_policy.stack.filter((_, itemIndex) => itemIndex !== index) : [])}>Remove LoRA</button>
                </fieldset>
              ))}
              <button
                type="button"
                className="secondary compact-button"
                disabled={!installedLoras.some((asset) => value.lora_policy.mode === "fixed" && !value.lora_policy.stack.some((lora) => lora.sha256 === asset.sha256))}
                onClick={() => {
                  if (value.lora_policy.mode !== "fixed") return;
                  const used = new Set(value.lora_policy.stack.map((lora) => lora.sha256));
                  const asset = installedLoras.find((candidate) => !used.has(candidate.sha256));
                  if (asset) updateStack([...value.lora_policy.stack, { sha256: asset.sha256, model_strength: asset.modelStrength, clip_strength: asset.clipStrength }]);
                }}
              >
                <Plus size={14} /> Add LoRA
              </button>
              {assets.isError && <p className="muted">Installed LoRAs could not be loaded.</p>}
            </div>
          )}
        </>
      )}
    </section>
  );
}
