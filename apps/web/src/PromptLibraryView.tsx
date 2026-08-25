import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Archive, ArchiveRestore, BookOpen, History, Pencil, Plus, Trash2 } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api, ApiError } from "./api";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { useConfirm } from "./useConfirm";
import { servesCapability } from "./workflowFamilies";
import type {
  PromptTemplateContract,
  PromptTemplateDetail,
  PromptTemplateLora,
  PromptTemplateLoraPolicy,
  PromptTemplateOptionLoraPolicy,
  PromptTemplateResourceOption,
  PromptTemplateResourcePolicy,
  PromptTemplateSlot,
  PromptTemplateSlotMode,
  PromptTemplateVariationScope,
} from "./types";

const PAGE_LIMIT = 50;
const SLOT_NAME = /^[a-z][a-z0-9_]{0,63}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const TOKEN = /{{([a-z][a-z0-9_]{0,63})}}/g;
const emptyLora = (): PromptTemplateLora => ({
  sha256: "",
  model_strength: 1,
  clip_strength: 1,
});

interface TemplateDraft {
  name: string;
  description: string;
  contract: PromptTemplateContract;
}

const defaultContract = (): PromptTemplateContract => ({
  schema_version: 1,
  operation: "text_to_image",
  body: "A {{subject}}.",
  slots: [{ name: "subject", mode: "input", variation_scope: "item" }],
  resource_policy: { mode: "inherited" },
});

const emptyDraft = (): TemplateDraft => ({
  name: "",
  description: "",
  contract: defaultContract(),
});

function templateSaveError(error: unknown): string | null {
  if (!error) return null;
  if (error instanceof ApiError && error.code === "prompt-template-name-taken") {
    return "A template with this name already exists. Choose a different name.";
  }
  if (error instanceof ApiError && error.code === "prompt-template-stale") {
    return "This template changed after you opened it. Close the editor, refresh, and try again.";
  }
  if (error instanceof ApiError && error.code === "prompt-template-resources-unavailable") {
    return "One or more selected workflows or LoRAs are no longer available.";
  }
  if (error instanceof ApiError && error.code === "prompt-template-request-invalid") {
    return "Review the template name, body, slots, and image resources, then try again.";
  }
  return "The Prompt Library could not save that revision. Review the template and try again.";
}

function draftFromTemplate(template: PromptTemplateDetail): TemplateDraft {
  return {
    name: template.name,
    description: template.description,
    contract: structuredClone(template.current_revision.contract_json),
  };
}

function slotForMode(
  current: PromptTemplateSlot,
  mode: PromptTemplateSlotMode,
): PromptTemplateSlot {
  const common = {
    name: current.name,
    variation_scope: mode === "fixed" ? "batch" as const : current.variation_scope,
  };
  if (mode === "choice") return {
    ...common,
    mode,
    choices: ["Option one", "Option two"],
    choice_strategy: "with_replacement",
  };
  if (mode === "model") return { ...common, mode, guidance: "Describe the requested variation." };
  if (mode === "fixed") return { ...common, mode, fixed_value: "Fixed text" };
  return { ...common, mode };
}

function validLoraStack(stack: PromptTemplateLora[], label: string): string | null {
  if (!stack.length || stack.length > 16) return `${label} needs between 1 and 16 LoRAs.`;
  const seen = new Set<string>();
  for (const lora of stack) {
    if (!SHA256.test(lora.sha256) || seen.has(lora.sha256)) {
      return `Choose an installed LoRA for ${label}.`;
    }
    seen.add(lora.sha256);
    if (![lora.model_strength, lora.clip_strength].every(
      (strength) => Number.isFinite(strength) && strength >= -4 && strength <= 4,
    )) return "LoRA strengths must be between -4 and 4.";
  }
  return null;
}

function validContract(contract: PromptTemplateContract): string | null {
  if (!contract.body.trim() || contract.body.length > 16_000) {
    return "Enter a template body of at most 16,000 characters.";
  }
  if (contract.slots.length > 32) return "A template can declare at most 32 slots.";
  const matches = [...contract.body.matchAll(TOKEN)];
  const bodyWithoutTokens = contract.body.replace(TOKEN, "");
  if (bodyWithoutTokens.includes("{") || bodyWithoutTokens.includes("}")) {
    return "Use only exact {{slot_name}} placeholders in the template body.";
  }
  const names = contract.slots.map((slot) => slot.name);
  if (names.some((name) => !SLOT_NAME.test(name)) || new Set(names).size !== names.length) {
    return "Slot names must be unique lowercase identifiers.";
  }
  if (matches.map((match) => match[1]).join("\0") !== names.join("\0")) {
    return "Place every declared slot exactly once, in the same order as the slot list.";
  }
  let totalChoices = 0;
  for (const slot of contract.slots) {
    if (slot.mode === "choice") {
      totalChoices += slot.choices.length;
      if (!slot.choices.length || slot.choices.length > 64) {
        return "Choice slots need between 1 and 64 choices.";
      }
      if (slot.choices.some((choice) => !choice.trim()) || new Set(slot.choices).size !== slot.choices.length) {
        return "Choice values must be non-empty and unique.";
      }
    }
    if (slot.mode === "model" && !slot.guidance.trim()) return "Model slots need guidance.";
    if (slot.mode === "fixed" && (!slot.fixed_value.trim() || slot.variation_scope !== "batch")) {
      return "Fixed slots need a value and must use batch scope.";
    }
  }
  if (totalChoices > 512) return "A template can contain at most 512 choices in total.";
  const resources = contract.resource_policy;
  if (resources.mode === "fixed") {
    if (!resources.workflow_revision_id.trim()) return "Choose an exact workflow revision.";
    if (resources.lora_policy.mode === "fixed") {
      return validLoraStack(resources.lora_policy.stack, "A fixed LoRA stack");
    }
    if (resources.lora_policy.mode === "pool") {
      const { stacks } = resources.lora_policy;
      if (stacks.length < 2 || stacks.length > 32) {
        return "A LoRA pool needs between 2 and 32 stacks.";
      }
      if (stacks.reduce((total, stack) => total + stack.length, 0) > 64) {
        return "A LoRA pool can contain at most 64 LoRAs in total.";
      }
      for (const [index, stack] of stacks.entries()) {
        const stackError = validLoraStack(stack, `Pool stack ${index + 1}`);
        if (stackError) return stackError;
      }
      if (new Set(stacks.map((stack) => JSON.stringify(stack))).size !== stacks.length) {
        return "Every LoRA pool stack must be unique.";
      }
    }
  }
  if (resources.mode === "pool") {
    // The editor already prevents both counts below - Remove option disables at
    // two, Add option at sixteen, and both Add LoRA and the fixed-stack choice
    // once sixty-four are pooled - so neither branch is reachable by authoring.
    // They are the second layer for a draft loaded from a contract that already
    // breaks the bound, which is the only way one can arrive here, and they keep
    // the browser refusal identical to the server rule rather than adjacent.
    const { options } = resources;
    if (options.length < 2 || options.length > 16) {
      return "A workflow pool needs between 2 and 16 options.";
    }
    let pooledLoras = 0;
    for (const [index, option] of options.entries()) {
      if (!option.workflow_revision_id.trim()) {
        return `Option ${index + 1} needs an exact workflow revision.`;
      }
      if (option.lora_policy.mode === "fixed") {
        const stackError = validLoraStack(option.lora_policy.stack, `Option ${index + 1}`);
        if (stackError) return stackError;
        pooledLoras += option.lora_policy.stack.length;
      }
    }
    if (pooledLoras > 64) {
      return "A workflow pool can pair at most 64 LoRAs in total.";
    }
    if (new Set(options.map(bundleKey)).size !== options.length) {
      return "Every workflow pool option must be a distinct workflow and LoRA bundle.";
    }
  }
  return null;
}

/** The server keys option uniqueness on the whole workflow-plus-LoRA bundle, not
 * on the revision alone, so one revision may appear twice with different LoRA
 * policies. Built as canonical JSON rather than a delimited join so no field can
 * impersonate a separator. */
function bundleKey(option: PromptTemplateResourceOption): string {
  const policy = option.lora_policy;
  const stack = policy.mode === "fixed"
    ? policy.stack.map((lora) => [lora.sha256, lora.model_strength, lora.clip_strength])
    : [];
  return JSON.stringify([option.workflow_revision_id, policy.mode, stack]);
}

function emptyPoolOption(): PromptTemplateResourceOption {
  return { workflow_revision_id: "", lora_policy: { mode: "inherited_auto" } };
}

function TemplateEditor({
  initial,
  creating,
  saving,
  requestError,
  onCancel,
  onSave,
}: {
  initial: TemplateDraft;
  creating: boolean;
  saving: boolean;
  requestError: unknown;
  onCancel: () => void;
  onSave: (draft: TemplateDraft) => void;
}) {
  const [draft, setDraft] = useState(() => structuredClone(initial));
  const [slotKeys, setSlotKeys] = useState(() => initial.contract.slots.map(() => crypto.randomUUID()));
  const [error, setError] = useState<string | null>(null);
  const contract = draft.contract;
  const replaceContract = (next: PromptTemplateContract) => setDraft((current) => ({
    ...current,
    contract: next,
  }));
  const replaceSlot = (index: number, slot: PromptTemplateSlot) => replaceContract({
    ...contract,
    slots: contract.slots.map((current, currentIndex) => currentIndex === index ? slot : current),
  });
  const submit = () => {
    if (!draft.name.trim() || draft.name.length > 200) {
      setError("Enter a template name of at most 200 characters.");
      return;
    }
    if (draft.description.length > 4_000) {
      setError("The description must be at most 4,000 characters.");
      return;
    }
    const contractError = validContract(contract);
    if (contractError) {
      setError(contractError);
      return;
    }
    onSave({ ...draft, name: draft.name.trim() });
  };
  const resources = contract.resource_policy;
  return (
    <AccessibleDialog
      title={creating ? "New prompt template" : `Edit ${initial.name}`}
      eyebrow="Prompt library"
      closeLabel="Close template editor"
      className="prompt-template-editor"
      onClose={onCancel}
    >
      <ErrorCallout message={error ?? templateSaveError(requestError)} />
      <label>
        Name
        <input aria-label="Template name" value={draft.name} placeholder="Name this template" maxLength={200} onChange={(event) => setDraft({ ...draft, name: event.target.value })} />
      </label>
      <label>
        Description
        <textarea aria-label="Template description" value={draft.description} maxLength={4_000} rows={2} onChange={(event) => setDraft({ ...draft, description: event.target.value })} />
      </label>
      <label>
        Template body
        <textarea aria-label="Template body" value={contract.body} maxLength={16_000} rows={5} onChange={(event) => replaceContract({ ...contract, body: event.target.value })} />
        <small>Place each slot once with an exact token such as {"{{subject}}"}.</small>
      </label>
      <section className="prompt-slot-editor" aria-labelledby="prompt-slots-heading">
        <div className="section-heading compact-heading">
          <div><h3 id="prompt-slots-heading">Slots</h3><p>Declaration order must match the body.</p></div>
          <button type="button" className="secondary compact-button" disabled={contract.slots.length >= 32} onClick={() => {
            const name = `slot_${contract.slots.length + 1}`;
            setSlotKeys((current) => [...current, crypto.randomUUID()]);
            replaceContract({
              ...contract,
              body: `${contract.body}${contract.body.endsWith(" ") ? "" : " "}{{${name}}}`,
              slots: [...contract.slots, { name, mode: "input", variation_scope: "item" }],
            });
          }}><Plus size={14} />Add slot</button>
        </div>
        {contract.slots.length === 0 && <p className="muted">This template has no variable slots.</p>}
        {contract.slots.map((slot, index) => (
          <fieldset className="prompt-slot-card" key={slotKeys[index]}>
            <legend>Slot {index + 1}</legend>
            <div className="prompt-slot-grid">
              <label>Name<input aria-label={`Slot ${index + 1} name`} value={slot.name} maxLength={64} onChange={(event) => replaceSlot(index, { ...slot, name: event.target.value })} /></label>
              <label>Mode<select aria-label={`Slot ${index + 1} mode`} value={slot.mode} onChange={(event) => replaceSlot(index, slotForMode(slot, event.target.value as PromptTemplateSlotMode))}><option value="input">Input</option><option value="choice">Choice</option><option value="model">Model-guided</option><option value="fixed">Fixed</option></select></label>
              <label>Variation<select aria-label={`Slot ${index + 1} variation`} disabled={slot.mode === "fixed"} value={slot.variation_scope} onChange={(event) => replaceSlot(index, { ...slot, variation_scope: event.target.value as PromptTemplateVariationScope })}><option value="item">Per item</option><option value="batch">Per batch</option></select></label>
              <button type="button" className="icon-button prompt-slot-remove" aria-label={`Remove slot ${index + 1}`} onClick={() => {
                setSlotKeys((current) => current.filter((_, currentIndex) => currentIndex !== index));
                replaceContract({ ...contract, slots: contract.slots.filter((_, currentIndex) => currentIndex !== index) });
              }}><Trash2 size={15} /></button>
            </div>
            {slot.mode === "choice" && <>
              <label>Choices, one per line<textarea aria-label={`Slot ${index + 1} choices`} rows={3} value={slot.choices.join("\n")} onChange={(event) => replaceSlot(index, { ...slot, choices: event.target.value.split("\n") })} /></label>
              <label>Choice use<select aria-label={`Slot ${index + 1} choice use`} value={slot.choice_strategy ?? "distinct"} onChange={(event) => replaceSlot(index, { ...slot, choice_strategy: event.target.value as "distinct" | "with_replacement" })}><option value="with_replacement">Allow repeats (recommended)</option><option value="distinct">Require distinct prompts</option></select></label>
            </>}
            {slot.mode === "model" && <label>Model guidance<textarea aria-label={`Slot ${index + 1} guidance`} rows={3} value={slot.guidance} maxLength={4_000} onChange={(event) => replaceSlot(index, { ...slot, guidance: event.target.value })} /></label>}
            {slot.mode === "fixed" && <label>Fixed value<input aria-label={`Slot ${index + 1} fixed value`} value={slot.fixed_value} maxLength={2_000} onChange={(event) => replaceSlot(index, { ...slot, fixed_value: event.target.value })} /></label>}
          </fieldset>
        ))}
      </section>
      <section className="prompt-resource-editor" aria-labelledby="prompt-resources-heading">
        <div className="section-heading compact-heading"><div><h3 id="prompt-resources-heading">Image setup</h3><p>Use this chat's setup, or choose saved workflows and LoRAs by name.</p></div></div>
        <label>Resource policy<select aria-label="Resource policy" value={resources.mode} onChange={(event) => {
          const mode = event.target.value;
          replaceContract({
            ...contract,
            resource_policy: mode === "fixed"
              ? { mode: "fixed", workflow_revision_id: "", lora_policy: { mode: "inherited_auto" } }
              : mode === "pool"
                ? { mode: "pool", strategy: "round_robin", options: [emptyPoolOption(), emptyPoolOption()] }
                : { mode: "inherited" },
          });
        }}><option value="inherited">Use this chat's setup (recommended)</option><option value="fixed">Always use chosen resources</option><option value="pool">Rotate resource sets (advanced)</option></select></label>
        {resources.mode === "fixed" && <FixedResourceEditor resources={resources} onChange={(resource_policy) => replaceContract({ ...contract, resource_policy })} />}
        {resources.mode === "pool" && <WorkflowPoolEditor resources={resources} onChange={(resource_policy) => replaceContract({ ...contract, resource_policy })} />}
      </section>
      <footer>
        <button type="button" className="secondary" onClick={onCancel}>Cancel</button>
        <button type="button" className="primary" disabled={saving} onClick={submit}>{saving ? "Saving…" : "Save revision"}</button>
      </footer>
    </AccessibleDialog>
  );
}

type ReadyWorkflowChoice = { id: string; label: string };

function useReadyImageWorkflows(): {
  choices: ReadyWorkflowChoice[];
  query: ReturnType<typeof useQuery<import("./types").WorkflowFamily[]>>;
} {
  const query = useQuery({
    queryKey: ["workflow-families", "image"],
    queryFn: () => api.workflowFamilies("image"),
  });
  const choices = useMemo(() => {
    const revisions: ReadyWorkflowChoice[] = [];
    const seen = new Set<string>();
    for (const family of query.data ?? []) {
      if (!servesCapability(family, "image")) continue;
      for (const variant of family.variants) {
        if (!variant.current_revision_id || variant.readiness !== "ready") continue;
        if (variant.operation !== "text_to_image") continue;
        if (seen.has(variant.current_revision_id)) continue;
        seen.add(variant.current_revision_id);
        revisions.push({
          id: variant.current_revision_id,
          label: `${family.name} - ${variant.name}${variant.current_revision_version ? ` - revision ${variant.current_revision_version}` : ""}`,
        });
      }
    }
    return revisions;
  }, [query.data]);
  return { choices, query };
}

function installedLoraDigest(asset: { manifest_json: Record<string, unknown> }): string | null {
  const value = asset.manifest_json.sha256;
  return typeof value === "string" && SHA256.test(value) ? value : null;
}
function FixedResourceEditor({
  resources,
  onChange,
}: {
  resources: Extract<PromptTemplateResourcePolicy, { mode: "fixed" }>;
  onChange: (resources: Extract<PromptTemplateResourcePolicy, { mode: "fixed" }>) => void;
}) {
  const policy = resources.lora_policy;
  const workflows = useReadyImageWorkflows();
  const [stackKeys, setStackKeys] = useState(() => policy.mode === "pool"
    ? policy.stacks.map(() => crypto.randomUUID())
    : []);
  const poolLoraCount = policy.mode === "pool"
    ? policy.stacks.reduce((total, stack) => total + stack.length, 0)
    : 0;
  return <>
    {workflows.query.isPending && <p className="prompt-pool-count">Checking ready image workflows...</p>}
    {workflows.query.isError && <p className="prompt-pool-count">Workflows could not be loaded. <button type="button" className="secondary compact-button" onClick={() => void workflows.query.refetch()}>Retry</button></p>}
    <label>Workflow<select aria-label="Workflow" value={resources.workflow_revision_id} onChange={(event) => onChange({ ...resources, workflow_revision_id: event.target.value })}>
      <option value="">Choose a ready image workflow</option>
      {workflows.choices.map((workflow) => <option key={workflow.id} value={workflow.id}>{workflow.label}</option>)}
      {resources.workflow_revision_id && !workflows.choices.some((workflow) => workflow.id === resources.workflow_revision_id)
        && <option value={resources.workflow_revision_id}>Previously selected workflow (currently unavailable)</option>}
    </select></label>
    <label>LoRA policy<select aria-label="LoRA policy" value={policy.mode} onChange={(event) => {
      const mode = event.target.value as PromptTemplateLoraPolicy["mode"];
      if (mode === "fixed") {
        setStackKeys([]);
        onChange({ ...resources, lora_policy: { mode, stack: [emptyLora()] } });
      } else if (mode === "pool") {
        setStackKeys([crypto.randomUUID(), crypto.randomUUID()]);
        onChange({
          ...resources,
          lora_policy: {
            mode,
            strategy: "round_robin",
            stacks: [[emptyLora()], [emptyLora()]],
          },
        });
      } else {
        setStackKeys([]);
        onChange({ ...resources, lora_policy: { mode } });
      }
    }}><option value="inherited_auto">Automatic LoRAs (recommended)</option><option value="none">No LoRAs</option><option value="fixed">Choose LoRAs</option><option value="pool">Rotate LoRA sets (advanced)</option></select></label>
    {policy.mode === "fixed" && <LoraStackEditor
      stack={policy.stack}
      ariaPrefix="LoRA"
      onChange={(stack) => onChange({ ...resources, lora_policy: { ...policy, stack } })}
    />}
    {policy.mode === "pool" && <div className="prompt-lora-editor">
      <label>Pool order<select aria-label="LoRA pool strategy" value={policy.strategy} onChange={(event) => onChange({
        ...resources,
        lora_policy: { ...policy, strategy: event.target.value as "random" | "round_robin" },
      })}><option value="round_robin">Round robin</option><option value="random">Deterministic random</option></select></label>
      <small>Each draft freezes one exact stack from this pool when it is queued.</small>
      {policy.stacks.map((stack, index) => <section className="prompt-lora-stack" key={stackKeys[index]} aria-labelledby={`prompt-lora-stack-${index + 1}`}>
        <div className="section-heading compact-heading"><h4 id={`prompt-lora-stack-${index + 1}`}>Stack {index + 1}</h4><button type="button" className="secondary compact-button" disabled={policy.stacks.length <= 2} onClick={() => {
          setStackKeys((current) => current.filter((_, currentIndex) => currentIndex !== index));
          onChange({ ...resources, lora_policy: { ...policy, stacks: policy.stacks.filter((_, stackIndex) => stackIndex !== index) } });
        }}>Remove stack</button></div>
        <LoraStackEditor
          stack={stack}
          ariaPrefix={`Stack ${index + 1} LoRA`}
          addDisabled={poolLoraCount >= 64}
          onChange={(next) => onChange({ ...resources, lora_policy: { ...policy, stacks: policy.stacks.map((item, stackIndex) => stackIndex === index ? next : item) } })}
        />
      </section>)}
      <button type="button" className="secondary compact-button" disabled={policy.stacks.length >= 32 || poolLoraCount >= 64} onClick={() => {
        setStackKeys((current) => [...current, crypto.randomUUID()]);
        onChange({ ...resources, lora_policy: { ...policy, stacks: [...policy.stacks, [emptyLora()]] } });
      }}><Plus size={14} />Add stack</button>
    </div>}
  </>;
}

function LoraStackEditor({
  stack,
  ariaPrefix,
  addDisabled = false,
  onChange,
}: {
  stack: PromptTemplateLora[];
  ariaPrefix: string;
  addDisabled?: boolean;
  onChange: (stack: PromptTemplateLora[]) => void;
}) {
  const [loraKeys, setLoraKeys] = useState(() => stack.map(() => crypto.randomUUID()));
  const assets = useQuery({
    queryKey: ["model-assets", "lora"],
    queryFn: () => api.modelAssets("lora"),
  });
  const installed = useMemo(() => (assets.data ?? []).flatMap((asset) => {
    const sha256 = installedLoraDigest(asset);
    return asset.active && asset.verified_at && sha256 ? [{
      sha256,
      name: asset.name,
      family: asset.family,
      modelStrength: asset.default_model_strength,
      clipStrength: asset.default_clip_strength,
    }] : [];
  }), [assets.data]);
  const used = new Set(stack.map((item) => item.sha256));
  const addable = installed.find((asset) => !used.has(asset.sha256));
  const update = (index: number, next: PromptTemplateLora) => onChange(
    stack.map((item, itemIndex) => itemIndex === index ? next : item),
  );
  return <div className="prompt-lora-editor">
    {assets.isPending && <p className="prompt-pool-count">Checking installed LoRAs...</p>}
    {assets.isError && <p className="prompt-pool-count">Installed LoRAs could not be loaded. <button type="button" className="secondary compact-button" onClick={() => void assets.refetch()}>Retry</button></p>}
    {stack.map((lora, index) => {
      const selected = installed.find((asset) => asset.sha256 === lora.sha256);
      return <fieldset key={loraKeys[index]}><legend>LoRA {index + 1}</legend><label>Installed LoRA<select aria-label={`${ariaPrefix} ${index + 1}`} value={lora.sha256} onChange={(event) => {
        const asset = installed.find((candidate) => candidate.sha256 === event.target.value);
        if (asset) update(index, { sha256: asset.sha256, model_strength: asset.modelStrength, clip_strength: asset.clipStrength });
      }}>
        {!selected && <option value={lora.sha256}>{lora.sha256 ? "Previously selected LoRA (currently unavailable)" : "Choose an installed LoRA"}</option>}
        {installed.map((asset) => <option key={asset.sha256} value={asset.sha256}>{asset.name}{asset.family ? ` - ${asset.family}` : ""}</option>)}
      </select></label><details><summary>Adjust strength</summary><div className="prompt-strength-grid"><label>Model strength<input aria-label={`${ariaPrefix} ${index + 1} model strength`} type="number" min={-4} max={4} step="0.05" value={lora.model_strength} onChange={(event) => update(index, { ...lora, model_strength: event.target.valueAsNumber })} /></label><label>CLIP strength<input aria-label={`${ariaPrefix} ${index + 1} CLIP strength`} type="number" min={-4} max={4} step="0.05" value={lora.clip_strength} onChange={(event) => update(index, { ...lora, clip_strength: event.target.valueAsNumber })} /></label></div></details><button type="button" className="secondary compact-button" onClick={() => {
      setLoraKeys((current) => current.filter((_, currentIndex) => currentIndex !== index));
      onChange(stack.filter((_, itemIndex) => itemIndex !== index));
    }}>Remove LoRA</button></fieldset>})}
    <button type="button" className="secondary compact-button" disabled={stack.length >= 16 || addDisabled || !addable} onClick={() => {
      if (!addable) return;
      setLoraKeys((current) => [...current, crypto.randomUUID()]);
      onChange([...stack, { sha256: addable.sha256, model_strength: addable.modelStrength, clip_strength: addable.clipStrength }]);
    }}><Plus size={14} />Add LoRA</button>
  </div>;
}

function WorkflowPoolEditor({
  resources,
  onChange,
}: {
  resources: Extract<PromptTemplateResourcePolicy, { mode: "pool" }>;
  onChange: (resources: Extract<PromptTemplateResourcePolicy, { mode: "pool" }>) => void;
}) {
  const { options } = resources;
  const [optionKeys, setOptionKeys] = useState(() => options.map(() => crypto.randomUUID()));
  // The bounded source of workflow identity the client already has. No API
  // surface is added for this: a pool option may only name a revision that is
  // current and ready, which is the same set the workflow selector offers.
  // The capability argument joins an image preference but does not require it
  // to be enabled, so the same `servesCapability` check the workflow selector
  // applies is applied here - otherwise a family the user has turned off for
  // image still contributes revisions, and the claim that this is the set the
  // selector offers would be false.
  //
  // The capability also filters FAMILIES rather than variants, and one family
  // can carry both text_to_image and image_to_image, so the operation is
  // filtered too. Prompt Library templates are text-to-image only.
  const families = useQuery({
    queryKey: ["workflow-families", "image"],
    queryFn: () => api.workflowFamilies("image"),
  });
  const readyRevisions = useMemo(() => {
    const revisions: { id: string; label: string }[] = [];
    const seen = new Set<string>();
    for (const family of families.data ?? []) {
      if (!servesCapability(family, "image")) continue;
      for (const variant of family.variants) {
        if (!variant.current_revision_id || variant.readiness !== "ready") continue;
        if (variant.operation !== "text_to_image") continue;
        if (seen.has(variant.current_revision_id)) continue;
        seen.add(variant.current_revision_id);
        revisions.push({
          id: variant.current_revision_id,
          label: `${family.name} · ${variant.name} (ready)`,
        });
      }
    }
    return revisions;
  }, [families.data]);
  const pooledLoras = options.reduce(
    (total, option) => total + (option.lora_policy.mode === "fixed" ? option.lora_policy.stack.length : 0),
    0,
  );
  const replaceOption = (index: number, next: PromptTemplateResourceOption) => onChange({
    ...resources,
    options: options.map((option, optionIndex) => optionIndex === index ? next : option),
  });
  return <>
    {families.isPending && <p className="prompt-pool-count">Checking which image workflows are ready…</p>}
    {families.isError && <p className="prompt-pool-count">Could not read which image workflows are ready. Pinned workflows are kept as they are. <button type="button" className="secondary compact-button" onClick={() => void families.refetch()}>Retry</button></p>}
    <label>Pool strategy<select aria-label="Pool strategy" value={resources.strategy} onChange={(event) => onChange({ ...resources, strategy: event.target.value as "random" | "round_robin" })}><option value="round_robin">Round robin by draft</option><option value="random">Deterministic random</option></select></label>
    <p className="prompt-pool-count">{options.length} option{options.length === 1 ? "" : "s"} · {pooledLoras} paired LoRA{pooledLoras === 1 ? "" : "s"} of 64</p>
    {options.map((option, index) => <fieldset key={optionKeys[index]} className="prompt-pool-option"><legend>Option {index + 1}</legend>
      <label>Workflow revision<select aria-label={`Option ${index + 1} workflow revision`} value={option.workflow_revision_id} onChange={(event) => replaceOption(index, { ...option, workflow_revision_id: event.target.value })}>
        <option value="">Choose a ready image workflow</option>
        {readyRevisions.map((revision) => <option key={revision.id} value={revision.id}>{revision.label}</option>)}
        {/* A template authored earlier can pin a revision that is no longer
          current or ready. Keep it selectable and say so, rather than
          silently moving the template onto today's tip.

          Only say it once the read succeeded. While the query is loading or
          failed the ready set is empty, and calling a pinned revision stale on
          that basis would turn an unknown read state into a false claim about
          the workflow. */}
        {option.workflow_revision_id && !readyRevisions.some((revision) => revision.id === option.workflow_revision_id)
          && <option value={option.workflow_revision_id}>{families.isSuccess ? "Previously selected workflow (currently unavailable)" : "Previously selected workflow"}</option>}
      </select></label>
      <label>LoRA policy<select aria-label={`Option ${index + 1} LoRA policy`} value={option.lora_policy.mode} onChange={(event) => {
        const mode = event.target.value as PromptTemplateOptionLoraPolicy["mode"];
        if (mode === "fixed" && option.lora_policy.mode !== "fixed" && pooledLoras >= 64) return;
        replaceOption(index, {
          ...option,
          lora_policy: mode === "fixed" ? { mode, stack: [emptyLora()] } : { mode },
        });
      }}><option value="inherited_auto">Automatic LoRAs (recommended)</option><option value="none">No LoRAs</option>{/* Choosing a fixed stack mints one LoRA, so at the cap this transition
        would author a sixty-fifth. Disabled rather than silently dropped, and
        guarded in the handler too because a disabled option can still be set
        programmatically. */}<option value="fixed" disabled={option.lora_policy.mode !== "fixed" && pooledLoras >= 64}>Choose LoRAs</option></select></label>
      {option.lora_policy.mode === "fixed" && <LoraStackEditor
        stack={option.lora_policy.stack}
        ariaPrefix={`Option ${index + 1} LoRA`}
        addDisabled={pooledLoras >= 64}
        onChange={(stack) => replaceOption(index, { ...option, lora_policy: { mode: "fixed", stack } })}
      />}
      <button type="button" className="secondary compact-button" disabled={options.length <= 2} onClick={() => {
        setOptionKeys((current) => current.filter((_, currentIndex) => currentIndex !== index));
        onChange({ ...resources, options: options.filter((_, optionIndex) => optionIndex !== index) });
      }}>Remove option</button>
    </fieldset>)}
    <button type="button" className="secondary compact-button" disabled={options.length >= 16} onClick={() => {
      setOptionKeys((current) => [...current, crypto.randomUUID()]);
      onChange({ ...resources, options: [...options, emptyPoolOption()] });
    }}><Plus size={14} />Add option</button>
  </>;
}

function resourceSummary(policy: PromptTemplateResourcePolicy): string {
  if (policy.mode === "inherited") return "Inherited image resources";
  if (policy.mode === "pool") {
    return `${policy.options.length}-option workflow pool · ${policy.strategy === "random" ? "random" : "round robin"}`;
  }
  if (policy.lora_policy.mode === "fixed") return `Fixed workflow · ${policy.lora_policy.stack.length} LoRA${policy.lora_policy.stack.length === 1 ? "" : "s"}`;
  if (policy.lora_policy.mode === "pool") return `Fixed workflow · ${policy.lora_policy.stacks.length}-stack LoRA pool`;
  return `Fixed workflow · ${policy.lora_policy.mode === "none" ? "No LoRAs" : "Automatic LoRAs"}`;
}

export function PromptLibraryView() {
  const client = useQueryClient();
  const [confirmDialog, confirm] = useConfirm();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [offset, setOffset] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editor, setEditor] = useState<{
    creating: boolean;
    draft: TemplateDraft;
    templateId: string | null;
    expectedRevisionId: string | null;
  } | null>(null);
  const page = useQuery({
    queryKey: ["prompt-templates", includeArchived, offset],
    queryFn: () => api.promptTemplates(includeArchived, PAGE_LIMIT, offset),
  });
  const resolvedSelectedId = selectedId && page.data?.items.some((item) => item.id === selectedId)
    ? selectedId
    : page.data?.items[0]?.id ?? null;
  const detail = useQuery({
    queryKey: ["prompt-template", resolvedSelectedId],
    queryFn: () => api.promptTemplate(resolvedSelectedId!),
    enabled: Boolean(resolvedSelectedId),
  });
  const revisions = useQuery({
    queryKey: ["prompt-template-revisions", resolvedSelectedId],
    queryFn: () => api.promptTemplateRevisions(resolvedSelectedId!),
    enabled: Boolean(resolvedSelectedId),
  });
  const refresh = () => {
    void client.invalidateQueries({ queryKey: ["prompt-templates"] });
    void client.invalidateQueries({ queryKey: ["prompt-template"] });
    void client.invalidateQueries({ queryKey: ["prompt-template-revisions"] });
  };
  const save = useMutation({
    mutationFn: ({ creating, draft, templateId, expectedRevisionId }: {
      creating: boolean;
      draft: TemplateDraft;
      templateId: string | null;
      expectedRevisionId: string | null;
    }) => {
      if (creating) {
        return api.createPromptTemplate({
          idempotency_key: crypto.randomUUID(),
          name: draft.name,
          description: draft.description,
          contract: draft.contract,
        });
      }
      if (!templateId || !expectedRevisionId) {
        throw new Error("Prompt template edit authority is unavailable.");
      }
      return api.updatePromptTemplate(templateId, {
        expected_current_revision_id: expectedRevisionId,
        idempotency_key: crypto.randomUUID(),
        name: draft.name,
        description: draft.description,
        contract: draft.contract,
      });
    },
    onSuccess: (result) => {
      setSelectedId(result.template.id);
      setEditor(null);
      refresh();
    },
  });
  const archive = useMutation({
    mutationFn: (template: PromptTemplateDetail) => api.updatePromptTemplate(template.id, {
      expected_current_revision_id: template.current_revision_id,
      archived: !template.archived,
    }),
    onSuccess: refresh,
  });
  const remove = useMutation({
    mutationFn: (template: PromptTemplateDetail) =>
      api.deletePromptTemplate(template.id, template.current_revision_id),
    onSuccess: () => {
      setSelectedId(null);
      setEditor(null);
      refresh();
    },
  });
  const confirmRemoval = async (template: PromptTemplateDetail) => {
    const confirmed = await confirm({
      title: `Delete ${template.name}?`,
      question: "Delete this prompt template from your library? This cannot be undone.",
      detail: <p>Immutable revisions and existing batch/import history will remain for provenance. The template cannot be edited, restored, exported, or used for new batches after deletion.</p>,
      confirmLabel: "Delete template",
      tone: "danger",
    });
    if (confirmed) remove.mutate(template);
  };
  const restore = useMutation({
    mutationFn: ({ template, revisionId }: { template: PromptTemplateDetail; revisionId: string }) =>
      api.restorePromptTemplateRevision(
        template.id,
        revisionId,
        template.current_revision_id,
        crypto.randomUUID(),
      ),
    onSuccess: refresh,
  });
  const selected = detail.data;
  const failure = page.error || detail.error || revisions.error
    || (!editor && save.error) || archive.error || restore.error || remove.error;
  const busy = save.isPending || archive.isPending || restore.isPending || remove.isPending;
  const currentSlots = selected?.current_revision.contract_json.slots ?? [];
  const archivedCount = useMemo(
    () => page.data?.items.filter((item) => item.archived).length ?? 0,
    [page.data],
  );
  return <div className="page-view prompt-library">
    <header className="page-header">
      <div><small>Reusable prompt structures</small><h1>Prompt library</h1><p>Author immutable image-template revisions with explicit inputs, controlled choices, and resource policies.</p></div>
      <button className="primary" onClick={() => { save.reset(); setEditor({ creating: true, draft: emptyDraft(), templateId: null, expectedRevisionId: null }); }}><Plus size={17} />New template</button>
    </header>
    <div className="prompt-library-toolbar">
      <label className="toggle-row"><span>Show archived templates</span><input type="checkbox" checked={includeArchived} onChange={(event) => { setIncludeArchived(event.target.checked); setOffset(0); }} /></label>
      {includeArchived && archivedCount > 0 && <small>{archivedCount} archived</small>}
      {page.data && (offset > 0 || page.data.total > PAGE_LIMIT) && <div className="prompt-page-controls"><button className="secondary compact-button" disabled={offset === 0 || page.isFetching} onClick={() => setOffset(Math.max(0, offset - PAGE_LIMIT))}>Previous</button><small>{page.data.items.length ? `${offset + 1}–${Math.min(offset + page.data.items.length, page.data.total)}` : "No items"} of {page.data.total}</small><button className="secondary compact-button" disabled={offset + page.data.items.length >= page.data.total || page.isFetching} onClick={() => setOffset(offset + PAGE_LIMIT)}>Next</button></div>}
    </div>
    <ErrorCallout message={failure instanceof ApiError ? failure.message : failure ? "The Prompt Library could not complete that request. Refresh and try again." : null} action={failure ? <button className="secondary compact-button" onClick={refresh}>Refresh</button> : undefined} />
    {page.isPending && <div className="loading-line" />}
    {!page.isPending && !page.data?.items.length ? <EmptyState icon={<BookOpen />} title="No prompt templates yet" body="Create a structured image prompt that can be revised without rewriting its history." /> : <div className="workflow-layout prompt-library-layout">
      <ul className="workflow-list prompt-template-list" aria-label="Prompt templates">
        {page.data?.items.map((template) => <li key={template.id}><button className={resolvedSelectedId === template.id ? "selected" : ""} onClick={() => setSelectedId(template.id)}><BookOpen size={17} /><span><strong>{template.name}</strong><small>{template.archived ? "Archived" : "Active"} · updated {new Date(template.updated_at).toLocaleDateString()}</small></span></button></li>)}
      </ul>
      <section className="workflow-detail prompt-template-detail" aria-live="polite">
        {detail.isPending && <div className="loading-line" />}
        {selected && <>
          <div className="detail-title"><div><small>Image template · revision {selected.current_revision.version}</small><h2>{selected.name}</h2></div><div className="storage-actions"><button className="secondary" disabled={busy} onClick={() => { save.reset(); setEditor({ creating: false, draft: draftFromTemplate(selected), templateId: selected.id, expectedRevisionId: selected.current_revision_id }); }}><Pencil size={14} />Edit</button><button className="secondary" disabled={busy} onClick={() => archive.mutate(selected)}>{selected.archived ? <ArchiveRestore size={14} /> : <Archive size={14} />}{selected.archived ? "Unarchive" : "Archive"}</button><button className="danger" disabled={busy} onClick={() => void confirmRemoval(selected)}><Trash2 size={14} />Delete</button></div></div>
          {selected.description && <p>{selected.description}</p>}
          <section className="prompt-template-body"><h3>Template body</h3><pre>{selected.current_revision.contract_json.body}</pre></section>
          <section className="prompt-template-slots"><h3>Slots</h3>{currentSlots.length ? <dl>{currentSlots.map((slot) => <div key={slot.name}><dt><code>{`{{${slot.name}}}`}</code><span className="badge">{slot.mode}</span><span className="badge">{slot.variation_scope}</span></dt><dd>{slot.mode === "choice" ? slot.choices.join(" · ") : slot.mode === "model" ? slot.guidance : slot.mode === "fixed" ? slot.fixed_value : "Provided when the template is used."}</dd></div>)}</dl> : <p className="muted">This template has no variable slots.</p>}</section>
          <p className="prompt-resource-summary">{resourceSummary(selected.current_revision.contract_json.resource_policy)}</p>
          <details className="prompt-template-history" open><summary><History size={14} />Revision history</summary>{revisions.data?.map((revision) => <div className="prompt-history-row" key={revision.id}><span><strong>Revision {revision.version}</strong><small>{new Date(revision.created_at).toLocaleString()}</small></span>{revision.id !== selected.current_revision_id && <button className="secondary compact-button" disabled={busy} onClick={() => restore.mutate({ template: selected, revisionId: revision.id })}>Restore</button>}</div>)}</details>
        </>}
      </section>
    </div>}
    {editor && <TemplateEditor initial={editor.draft} creating={editor.creating} saving={save.isPending} requestError={save.error} onCancel={() => { save.reset(); setEditor(null); }} onSave={(draft) => save.mutate({ creating: editor.creating, draft, templateId: editor.templateId, expectedRevisionId: editor.expectedRevisionId })} />}
    {confirmDialog}
  </div>;
}
