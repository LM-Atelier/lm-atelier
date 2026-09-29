import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { WorkflowRecipeChoices } from "./WorkflowRecipeChoices";
import { WorkflowRecipeEditor } from "./WorkflowRecipeEditor";
import { workflowRecipeCatalog } from "./workflowRecipeCatalog";
import { workflowUseCases } from "./workflowUseCaseTypes";
import type { WorkflowUseCasePreset, WorkflowUseCasePresetCreate } from "./workflowUseCaseTypes";
import "./WorkflowRecipeManager.css";

export function WorkflowRecipeManagerAction() {
  const [open, setOpen] = useState(false);
  return <><button type="button" className="secondary" onClick={() => setOpen(true)}>Manage recipes</button>
    {open && createPortal(<WorkflowRecipeManager onClose={() => setOpen(false)} />, document.body)}</>;
}

export function WorkflowRecipeManager({ onClose }: { onClose: () => void }) {
  const client = useQueryClient();
  const catalog = useQuery({ queryKey: ["workflow-recipes"], queryFn: ({ signal }) => workflowRecipeCatalog(signal) });
  const projects = useQuery({ queryKey: ["projects", "recipe-choices"], queryFn: () => api.projects() });
  const [editing, setEditing] = useState<WorkflowUseCasePreset | "new" | null>(null);
  const [deleting, setDeleting] = useState<WorkflowUseCasePreset | null>(null);
  const [projectId, setProjectId] = useState("");
  const [notice, setNotice] = useState("");
  const [defaultsOpen, setDefaultsOpen] = useState(false);
  const [projectsOpen, setProjectsOpen] = useState(false);
  const newRecipeButton = useRef<HTMLButtonElement>(null);
  const confirmDeletionButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (editing !== null) return;
    if (deleting !== null) confirmDeletionButton.current?.focus();
    else newRecipeButton.current?.focus();
  }, [editing, deleting]);
  const refresh = () => {
    void client.invalidateQueries({ queryKey: ["workflow-recipes"] });
    void client.invalidateQueries({ queryKey: ["workflow-recipe-choice"] });
  };
  const save = useMutation({
    mutationFn: ({ id, payload }: { id: string | null; payload: WorkflowUseCasePresetCreate }) => id
      ? api.replaceWorkflowUseCasePreset(id, payload) : api.createWorkflowUseCasePreset(payload),
    onSuccess: () => { setEditing(null); setNotice("Recipe saved."); refresh(); },
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteWorkflowUseCasePreset(id),
    onSuccess: () => { setDeleting(null); setNotice("Recipe deleted."); refresh(); },
  });
  const busy = save.isPending || remove.isPending;
  const project = projects.error ? undefined : projects.data?.find((item) => item.id === projectId);
  return <AccessibleDialog title="Workflow recipes" eyebrow="Reusable settings" closeLabel="Close recipe manager"
    onClose={() => { if (!busy) onClose(); }} className="workflow-recipe-manager">
    {editing !== null ? <WorkflowRecipeEditor key={editing === "new" ? "new" : editing.id}
      recipe={editing === "new" ? null : editing} saving={save.isPending} error={save.error}
      onSave={(payload) => { if (!busy) save.mutate({ id: editing === "new" ? null : editing.id, payload }); }}
      onCancel={() => { if (!busy) { setEditing(null); save.reset(); } }} /> : <>
      <p>Save settings for a use case, then choose where they apply. Explicit turn settings take priority.</p>
      <button ref={newRecipeButton} type="button" className="primary" aria-disabled={busy} onClick={() => { if (!busy) { setEditing("new"); setDeleting(null); setNotice(""); save.reset(); } }}>New recipe</button>
      {notice && <p role="status">{notice}</p>}
      {catalog.isPending && <p role="status">Loading recipes…</p>}
      {catalog.error && <p role="alert">{catalog.error.message}<button type="button" className="secondary compact-button" onClick={() => void catalog.refetch()}>Retry recipes</button></p>}
      {!catalog.error && catalog.data && <ul className="workflow-recipe-list">{catalog.data.map((recipe) => <li key={recipe.id}>
        <div><strong>{recipe.name}</strong><small>{workflowUseCases.find((item) => item.id === recipe.use_case)?.label}
          {recipe.builtin ? " · Built in" : ""}{!recipe.enabled ? " · Disabled" : ""}{recipe.is_default ? " · Workspace default" : ""}</small></div>
        {!recipe.builtin && <div className="row-actions"><button type="button" className="secondary compact-button" aria-label={`Edit recipe ${recipe.name}`}
          onClick={() => { if (!busy) { setEditing(structuredClone(recipe)); setDeleting(null); save.reset(); } }}>Edit</button>
          <button type="button" className="secondary compact-button" aria-label={`Delete recipe ${recipe.name}`}
            onClick={() => { if (!busy) { setDeleting(recipe); remove.reset(); } }}>Delete</button></div>}
      </li>)}</ul>}
      {!catalog.error && catalog.data?.length === 0 && <p>No recipes yet. Create one to save settings for a request type.</p>}
      {deleting && <div className="workflow-recipe-delete" role="group" aria-label="Confirm recipe deletion">
        <p>Delete {deleting.name}? Remove any active selections first.</p>
        {remove.error && <p role="alert">{remove.error.message}</p>}
        <button ref={confirmDeletionButton} type="button" className="secondary danger" aria-disabled={busy} onClick={() => { if (!busy) remove.mutate(deleting.id); }}>Confirm deletion</button>
        <button type="button" className="secondary" aria-disabled={busy} onClick={() => { if (!busy) setDeleting(null); }}>Keep recipe</button>
      </div>}
      <details open={defaultsOpen} onToggle={(event) => setDefaultsOpen(event.currentTarget.open)}><summary>Workspace defaults</summary>
        {defaultsOpen && <><p>These apply when neither the chat nor its project selects a recipe or Automatic.</p>
        <WorkflowRecipeChoices scope={{ kind: "workspace" }} /></>}</details>
      <details open={projectsOpen} onToggle={(event) => setProjectsOpen(event.currentTarget.open)}><summary>Project choices</summary>
        {projectsOpen && <>
        <label>Project for recipe choices<select value={projectId} disabled={projects.isPending || Boolean(projects.error)} onChange={(event) => setProjectId(event.target.value)}>
          <option value="">{projects.isPending ? "Loading projects…" : projects.error ? "Cannot read projects" : "Choose a project"}</option>
          {projectId && !project && <option value={projectId} disabled>Selected project unavailable</option>}
          {!projects.error && projects.data?.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select></label>
        {projects.error && <p role="alert">{projects.error.message}<button type="button" className="secondary compact-button" onClick={() => void projects.refetch()}>Retry projects</button></p>}
        {project && <WorkflowRecipeChoices scope={{ kind: "project", id: project.id }} />}
        </>}
      </details>
    </>}
  </AccessibleDialog>;
}
