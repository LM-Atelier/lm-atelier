import { useId, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { workflowRecipeCatalog } from "./workflowRecipeCatalog";
import { workflowUseCases } from "./workflowUseCaseTypes";
import type { WorkflowRecipeScope, WorkflowUseCase, WorkflowUseCaseChoice } from "./workflowUseCaseTypes";
import "./WorkflowRecipeChoices.css";

function scopeKey(scope: WorkflowRecipeScope) {
  return scope.kind === "workspace" ? "workspace" : `${scope.kind}:${scope.id}`;
}

function RecipeChoice({ scope, useCase, label }: {
  scope: WorkflowRecipeScope; useCase: WorkflowUseCase; label: string;
}) {
  const id = useId();
  const client = useQueryClient();
  const queryKey = ["workflow-recipe-choice", scopeKey(scope), useCase];
  const catalog = useQuery({ queryKey: ["workflow-recipes"], queryFn: ({ signal }) => workflowRecipeCatalog(signal) });
  const current = useQuery({
    queryKey,
    queryFn: async ({ signal }): Promise<WorkflowUseCaseChoice> => {
      if (scope.kind !== "workspace") return api.workflowUseCaseChoice(scope, useCase, signal);
      const value = await api.workflowUseCaseDefault(useCase, signal);
      return value.preset_id === null ? { mode: "automatic" } : { mode: "preset", preset_id: value.preset_id };
    },
  });
  const save = useMutation({
    onMutate: () => client.cancelQueries({ queryKey, exact: true }),
    mutationFn: async (choice: WorkflowUseCaseChoice): Promise<WorkflowUseCaseChoice> => {
      if (scope.kind !== "workspace") return api.setWorkflowUseCaseChoice(scope, useCase, choice);
      const value = await api.setWorkflowUseCaseDefault(useCase, { preset_id: choice.mode === "preset" ? choice.preset_id : null });
      return value.preset_id === null ? { mode: "automatic" } : { mode: "preset", preset_id: value.preset_id };
    },
    onSuccess: (value) => {
      client.setQueryData(queryKey, value);
      if (scope.kind === "workspace") void client.invalidateQueries({ queryKey: ["workflow-recipes"] });
    },
  });
  const error = current.error ?? catalog.error;
  const unresolved = Boolean(error) || !current.data || !catalog.data;
  const selectedId = current.data?.mode === "preset" ? current.data.preset_id : null;
  const available = (catalog.data ?? []).filter((recipe) => recipe.enabled && recipe.use_case === useCase);
  const unavailable = selectedId !== null && !available.some((recipe) => recipe.id === selectedId);
  const selected = catalog.data?.find((recipe) => recipe.id === selectedId);
  const value = unresolved ? "" : selectedId !== null ? `preset:${selectedId}` : current.data!.mode;
  return <div className="workflow-recipe-choice">
    <label htmlFor={id}>{label} recipe</label>
    <select id={id} value={value} disabled={unresolved} aria-disabled={save.isPending}
      aria-describedby={`${id}-status`} onChange={(event) => {
        if (unresolved || save.isPending) return;
        const next = event.target.value;
        if (next === "inherit" && scope.kind !== "workspace") save.mutate({ mode: "inherit" });
        else if (next === "automatic") save.mutate({ mode: "automatic" });
        else if (next.startsWith("preset:") && available.some((recipe) => recipe.id === next.slice(7))) {
          save.mutate({ mode: "preset", preset_id: next.slice(7) });
        }
      }}>
      {unresolved ? <option value="">{error ? "Cannot read recipes" : "Loading recipes…"}</option> : <>
        {scope.kind !== "workspace" && <option value="inherit">{scope.kind === "chat" ? "Inherit project / workspace" : "Inherit workspace"}</option>}
        <option value="automatic">Automatic (no recipe)</option>
        {unavailable && <option value={`preset:${selectedId}`} disabled>{selected?.name ?? "Selected recipe"} (unavailable)</option>}
        {available.map((recipe) => <option key={recipe.id} value={`preset:${recipe.id}`}>{recipe.name}</option>)}
      </>}
    </select>
    <div id={`${id}-status`} className="workflow-recipe-status">
      {error && <><small role="alert">{error.message}</small>
        <button type="button" className="secondary compact-button" aria-label={`Retry ${label} recipe`}
          onClick={() => { void current.refetch(); void catalog.refetch(); }}>Try again</button></>}
      {save.isPending && <small role="status">Saving…</small>}
      {save.error && <small role="alert">{save.error.message}</small>}
      {!unresolved && unavailable && <small role="status">{scope.kind === "workspace"
        ? "Choose an available recipe or Automatic before sending."
        : "Choose an available recipe, Automatic, or inheritance before sending."}</small>}
    </div>
  </div>;
}

export function WorkflowRecipeChoices({ scope }: { scope: WorkflowRecipeScope }) {
  return <div className="workflow-recipe-choices" role="group" aria-label="Use-case recipes">
    {workflowUseCases.map(({ id, label }) => <RecipeChoice key={`${scopeKey(scope)}:${id}`} scope={scope} useCase={id} label={label} />)}
  </div>;
}

export function ChatWorkflowRecipes({ chatId }: { chatId: string }) {
  const [open, setOpen] = useState(false);
  return <details className="chat-workflow-recipes" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Recipes for this chat</summary>
    {open && <div className="chat-workflow-recipe-body">
      <p>Recipes supply settings for each request type. Inherit uses project, then workspace choices.
        Automatic skips recipes. Workflow choices above still apply, and explicit turn settings take priority.</p>
      <WorkflowRecipeChoices scope={{ kind: "chat", id: chatId }} />
    </div>}
  </details>;
}
