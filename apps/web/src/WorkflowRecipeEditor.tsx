import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { SettingControl } from "./SettingControl";
import { useWorkflowRevisionSchema } from "./useWorkflowRevisionSchema";
import { isRecipeSettingValue, recipeFields, recipeOperation, recipeRole, recipeValueError } from "./workflowRecipeFields";
import { workflowUseCases } from "./workflowUseCaseTypes";
import type { RecipeSettingValue, WorkflowUseCase, WorkflowUseCasePreset, WorkflowUseCasePresetCreate } from "./workflowUseCaseTypes";
import "./WorkflowRecipeManager.css";

export function WorkflowRecipeEditor({ recipe, saving, error, onSave, onCancel }: {
  recipe: WorkflowUseCasePreset | null; saving: boolean; error: Error | null;
  onSave: (payload: WorkflowUseCasePresetCreate) => void; onCancel: () => void;
}) {
  const [name, setName] = useState(recipe?.name ?? "");
  const [useCase, setUseCase] = useState<WorkflowUseCase>(recipe?.use_case ?? "image_generation");
  const [enabled, setEnabled] = useState(recipe?.enabled ?? true);
  const [settings, setSettings] = useState<Record<string, RecipeSettingValue>>(() => structuredClone(recipe?.settings_json ?? {}));
  const [workflowId, setWorkflowId] = useState("");
  const nameField = useRef<HTMLInputElement>(null);
  useEffect(() => { nameField.current?.focus(); }, []);
  const workflows = useQuery({ queryKey: ["workflows", "summaries"], queryFn: () => api.workflowSummaries() });
  const engines = useQuery({ queryKey: ["engines"], queryFn: () => api.engines() });
  const operation = recipeOperation(useCase);
  const references = workflows.error ? [] : (workflows.data ?? []).filter((workflow) => workflow.operation === operation && workflow.current_revision_id);
  const reference = references.find((workflow) => workflow.id === workflowId);
  const revision = useWorkflowRevisionSchema(reference?.current_revision_id ?? null, operation);
  const engine = engines.error ? undefined : engines.data?.find((candidate) => candidate.roles.includes(recipeRole(useCase)));
  const readError = workflows.error ?? engines.error ?? revision.error;
  const fields = reference && revision.schema && engine && !readError ? recipeFields(engine, useCase, revision.schema) : [];
  const unavailable = Object.keys(settings).filter((key) => !fields.some((field) => field.key === key));
  const valueErrors = fields.flatMap((field) => Object.hasOwn(settings, field.key)
    ? [recipeValueError(field, settings[field.key])].filter((message): message is string => message !== null) : []);
  const ready = Boolean(reference && revision.schema && engine && !readError);
  const settingsChanged = JSON.stringify(settings) !== JSON.stringify(recipe?.settings_json ?? {}) || useCase !== (recipe?.use_case ?? useCase);
  const canSave = Boolean(name.trim()) && !saving && !valueErrors.length
    && (!Object.keys(settings).length || !settingsChanged || ready);
  function remove(key: string) {
    setSettings((current) => Object.fromEntries(Object.entries(current).filter(([name]) => name !== key)));
  }
  return <form className="workflow-recipe-editor" onSubmit={(event) => {
    event.preventDefault();
    if (!canSave || !event.currentTarget.reportValidity()) return;
    onSave({ name: name.trim(), use_case: useCase, settings_json: structuredClone(settings), enabled, is_default: recipe?.is_default ?? false });
  }}>
    <label>Recipe name<input ref={nameField} required maxLength={200} value={name} disabled={saving} onChange={(event) => setName(event.target.value)} /></label>
    <label>Use case<select value={useCase} disabled={saving} onChange={(event) => { setUseCase(event.target.value as WorkflowUseCase); setWorkflowId(""); }}>
      {workflowUseCases.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
    </select></label>
    <label><input type="checkbox" checked={enabled} disabled={saving || recipe?.is_default}
      onChange={(event) => setEnabled(event.target.checked)} />Enabled</label>
    {recipe?.is_default && <p>Choose another workspace default before disabling this recipe.</p>}
    <label>Reference workflow for settings<select value={workflowId} disabled={saving || workflows.isPending || Boolean(workflows.error)}
      onChange={(event) => setWorkflowId(event.target.value)}>
      <option value="">{workflows.isPending ? "Loading workflows…" : workflows.error ? "Cannot read workflows" : "Choose a reference workflow"}</option>
      {workflowId && !reference && <option value={workflowId} disabled>Selected reference unavailable</option>}
      {references.map((workflow) => <option key={workflow.id} value={workflow.id}>{workflow.name}</option>)}
    </select></label>
    <p>The reference supplies setting controls. Your workflow choices still determine execution;
      compatibility is checked against the selected revision when you send.</p>
    {readError && <div role="alert">{readError.message}<button type="button" className="secondary compact-button"
      onClick={() => { void workflows.refetch(); void engines.refetch(); if (reference) void revision.retry(); }}>Retry setting controls</button></div>}
    {!workflows.isPending && !workflows.error && references.length === 0 && <p>No workflow for this request type is installed. Add one from the workflow library to edit its settings.</p>}
    {reference && !ready && !readError && <p role="status">{engines.isPending || revision.isLoading ? "Loading setting controls…" : "Setting controls are unavailable for this reference."}</p>}
    {ready && fields.length === 0 && <p>This reference offers no editable recipe settings.</p>}
    <fieldset disabled={saving} className="workflow-recipe-fields"><legend>Included settings</legend>
      {fields.map((field) => <div key={`${reference?.current_revision_id}:${field.key}`} className="workflow-recipe-field">
        <label><input type="checkbox" checked={Object.hasOwn(settings, field.key)} onChange={(event) => {
          if (!event.target.checked) remove(field.key);
          else if (isRecipeSettingValue(field.default)) setSettings((current) => ({ ...current, [field.key]: structuredClone(field.default as RecipeSettingValue) }));
        }} />Include {field.label}</label>
        {Object.hasOwn(settings, field.key) && <SettingControl field={field.type === "integer" ? { ...field, type: "number", step: field.step ?? 1 } : field} value={settings[field.key]} onChange={(value) => {
          if (isRecipeSettingValue(value)) setSettings((current) => ({ ...current, [field.key]: value }));
        }} />}
      </div>)}
      {unavailable.length > 0 && <div><p>These saved settings are retained. Choose a reference that supports them or remove them explicitly.</p>
        <ul>{unavailable.map((key) => <li key={key}><span>{key} (not available in these controls)</span>
          <button type="button" className="secondary compact-button" onClick={() => remove(key)} aria-label={`Remove saved setting ${key}`}>Remove</button></li>)}</ul></div>}
    </fieldset>
    {valueErrors.map((message) => <p key={message} role="alert">{message}</p>)}
    {error && <p role="alert">{error.message}</p>}
    <div className="row-actions"><button type="submit" className="primary" aria-disabled={!canSave}>{saving ? "Saving recipe…" : "Save recipe"}</button>
      <button type="button" className="secondary" aria-disabled={saving} onClick={() => { if (!saving) onCancel(); }}>Cancel recipe changes</button></div>
  </form>;
}
