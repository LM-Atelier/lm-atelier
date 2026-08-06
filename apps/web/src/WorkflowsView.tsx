import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Workflow as WorkflowIcon } from "lucide-react";
import { api } from "./api";
import { AccessibleDialog } from "./AccessibleDialog";
import { CustomNodesPanel } from "./CustomNodesPanel";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { RegistryInstallsPanel } from "./RegistryInstallsPanel";
import { WorkflowFamilyPreferences } from "./WorkflowFamilyPreferences";
import { WorkflowPackageReview } from "./WorkflowPackageReview";
import { useWorkflowPackageImport } from "./useWorkflowPackageImport";
import { downloadJson } from "./format";

export function WorkflowControls({ schema }: { schema: Record<string, unknown> }) {
  const properties = schema.properties && typeof schema.properties === "object"
    ? schema.properties as Record<string, Record<string, unknown>>
    : {};
  const schemaKey = JSON.stringify(schema);
  const defaults = Object.fromEntries(
    Object.entries(properties).map(([key, field]) => [key, field.default ?? ""]),
  );
  const [stored, setStored] = useState<{ schemaKey: string; values: Record<string, unknown> }>(
    { schemaKey, values: defaults },
  );
  const values = stored.schemaKey === schemaKey ? stored.values : defaults;
  if (!Object.keys(properties).length) return <p className="muted">This revision does not declare user-facing inputs.</p>;
  return (
    <div className="workflow-controls">
      {Object.entries(properties).map(([key, field]) => {
        const label = String(field.title ?? key.replaceAll("_", " "));
        const description = typeof field.description === "string" ? field.description : "";
        const choices = Array.isArray(field.enum) ? field.enum : [];
        const type = String(field.type ?? "string");
        const update = (value: unknown) => setStored((current) => ({
          schemaKey,
          values: {
            ...(current.schemaKey === schemaKey ? current.values : defaults),
            [key]: value,
          },
        }));
        return (
          <label key={key}>
            <span><strong>{label}</strong>{description && <small>{description}</small>}</span>
            {choices.length ? (
              <select value={String(values[key] ?? "")} onChange={(event) => update(event.target.value)}>
                {choices.map((choice) => <option key={String(choice)} value={String(choice)}>{String(choice)}</option>)}
              </select>
            ) : type === "boolean" ? (
              <input type="checkbox" checked={Boolean(values[key])} onChange={(event) => update(event.target.checked)} />
            ) : (
              <input type={type === "integer" || type === "number" ? "number" : "text"} min={typeof field.minimum === "number" ? field.minimum : undefined} max={typeof field.maximum === "number" ? field.maximum : undefined} step={type === "integer" ? 1 : typeof field.multipleOf === "number" ? field.multipleOf : undefined} value={String(values[key] ?? "")} onChange={(event) => update(type === "integer" || type === "number" ? Number(event.target.value) : event.target.value)} />
            )}
          </label>
        );
      })}
    </div>
  );
}
export function WorkflowsView() {
  const client = useQueryClient();
  const workflows = useQuery({ queryKey: ["workflows"], queryFn: api.workflows }); const families = useQuery({ queryKey: ["workflow-families"], queryFn: () => api.workflowFamilies() });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = workflows.data?.find((workflow) => workflow.id === selectedId) ?? null; const selectedFamily = families.data?.find((family) => family.variants.some((variant) => variant.id === selectedId));
  const [selectedRevisionId, setSelectedRevisionId] = useState<string | null>(null);
  const [newOpen, setNewOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState("Custom image workflow");
  const [description, setDescription] = useState("");
  const [operation, setOperation] = useState("text_to_image");
  const [graph, setGraph] = useState("{}");
  const [uiGraph, setUiGraph] = useState("{}");
  const [inputSchema, setInputSchema] = useState("{}");
  const [dependencies, setDependencies] = useState("{}");
  const [trusted, setTrusted] = useState(false);
  const importInput = useRef<HTMLInputElement>(null);
  // Everything on this page that changes a workflow calls this: create, new
  // revision, duplicate, restore, and both import paths. It refreshed the
  // list alone, while the server derives a family's current revision, engine,
  // capabilities and readiness from the very revision that just changed - so
  // the families beside the list, the selectors elsewhere, and the studio's
  // idea of which tools exist all kept the previous answer.
  //
  // The keys are prefixes, so ["workflow-families"] also covers the
  // per-capability entries the selector holds.
  const refresh = () => {
    void client.invalidateQueries({ queryKey: ["workflows"] });
    void client.invalidateQueries({ queryKey: ["workflow-families"] });
    void client.invalidateQueries({ queryKey: ["studio-capabilities"] });
  };
  const save = useMutation({
    mutationFn: async () => {
      const revision = { engine_version: null, api_graph: JSON.parse(graph), ui_graph: JSON.parse(uiGraph), input_schema: JSON.parse(inputSchema), dependencies: JSON.parse(dependencies), trusted };
      if (editing && selected) {
        // Two writes, and the second is the one that validates: a rejected
        // schema or dependency block used to leave a committed rename behind
        // while the dialog reported one failed save. Skipping the write that
        // has nothing to say removes the split for the ordinary case, where
        // someone is editing the graph and not renaming anything.
        //
        // The order stays as it is. Creating the revision first would make it
        // current before the rename could fail, and the obvious retry would
        // then mint a second revision and quietly promote that instead.
        if (name !== selected.name || description !== selected.description) {
          await api.updateWorkflow(selected.id, { name, description });
        }
        return api.createWorkflowRevision(selected.id, revision);
      }
      return api.createWorkflow({ name, description, operation, engine: "comfyui", ...revision });
    },
    onSuccess: () => { setNewOpen(false); setEditing(false); },
    // Refreshed either way. A rename that did land while the revision was
    // refused is real, and leaving the list showing the old name hid it until
    // some later visit produced it with no action from the reader.
    onSettled: () => refresh(),
  });
  const validate = useMutation({ mutationFn: (id: string) => api.validateWorkflow(id) });
  const clone = useMutation({ mutationFn: (id: string) => api.cloneWorkflow(id), onSuccess: refresh });
  const restore = useMutation({ mutationFn: ({ id, revisionId }: { id: string; revisionId: string }) => api.restoreWorkflowRevision(id, revisionId), onSuccess: refresh });
  const exportBundle = useMutation({ mutationFn: (id: string) => api.exportWorkflow(id), onSuccess: (bundle) => downloadJson(bundle, `${bundle.name.replaceAll(/[^a-z0-9]+/gi, "-").toLowerCase()}.lm-atelier-workflow.json`) });
  // The graph downloads, and the way in is offered as a link rather than
  // opened for you. A window asked for after the click is a popup, which
  // browsers refuse - and `noopener` makes window.open return null whether it
  // was refused or not, so the old code could not have noticed either way.
  const [comfyTarget, setComfyTarget] = useState<{ url: string; filename: string } | null>(null);
  const openInComfy = useMutation({
    mutationFn: (id: string) => api.workflowOpenTarget(id),
    onSuccess: (target) => {
      // The link first: whatever the browser makes of the download, the way
      // in is already known and should not depend on it.
      setComfyTarget({ url: target.url, filename: target.filename });
      downloadJson(target.ui_graph, target.filename);
    },
  });
  const {
    importFile: importBundle,
    importError,
    packageReview,
    closePackageReview,
  } = useWorkflowPackageImport(refresh);
  const openCreate = () => { setEditing(false); setName("Custom image workflow"); setDescription(""); setOperation("text_to_image"); setGraph("{}"); setUiGraph("{}"); setInputSchema("{}"); setDependencies("{}"); setTrusted(false); setNewOpen(true); };
  const openEdit = () => { if (!selected) return; const revision = selected.revisions.find((item) => item.id === selected.current_revision_id) ?? selected.revisions.at(-1); if (!revision) return; setEditing(true); setName(selected.name); setDescription(selected.description); setOperation(selected.operation); setGraph(JSON.stringify(revision.api_graph_json, null, 2)); setUiGraph(JSON.stringify(revision.ui_graph_json, null, 2)); setInputSchema(JSON.stringify(revision.input_schema_json, null, 2)); setDependencies(JSON.stringify(revision.dependencies_json, null, 2)); setTrusted(revision.trusted); setNewOpen(true); };
  // A verdict is about the workflow that was validated. Held globally by
  // the mutation it stayed on screen when the selection moved, reading as
  // the new workflow's result - the right answer under the wrong name.
  const verdict = validate.data && validate.variables === selected?.id ? validate.data : null;
  const selectedRevision = selected?.revisions.find((revision) => revision.id === selectedRevisionId) ?? selected?.revisions.find((revision) => revision.id === selected.current_revision_id) ?? selected?.revisions.at(-1);
  const currentRevision = selected?.revisions.find((revision) => revision.id === selected.current_revision_id);
  return (
    <div className="page-view">
      <header className="page-header"><div><h1>Workflows</h1></div><div className="storage-actions"><input ref={importInput} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0]); event.target.value = ""; }} /><button className="secondary" onClick={() => importInput.current?.click()}>Import bundle</button><button className="primary" onClick={openCreate}><Plus size={17} />New workflow</button></div></header>
      {/* A list that could not be read is not an empty list, and a family
          list that failed is not "no preferences". Both used to render as
          the unselected state, which invites the reader to pick from
          nothing and tells them nothing went wrong. */}
      {(workflows.error || families.error) && (
        <ErrorCallout message={((workflows.error ?? families.error) as Error).message} />
      )}
      {(importError || clone.error || restore.error || exportBundle.error || openInComfy.error || validate.error) && <ErrorCallout message={(importError || clone.error || restore.error || exportBundle.error || openInComfy.error || validate.error)?.message} />}
      {packageReview && <WorkflowPackageReview analysis={packageReview.analysis} fileName={packageReview.fileName} uiGraph={packageReview.uiGraph} onImported={() => { closePackageReview(); refresh(); }} onClose={closePackageReview} />}
      {selected && (
        <div className="storage-actions">
          <button
            className="secondary"
            disabled={openInComfy.isPending}
            onClick={() => { setComfyTarget(null); openInComfy.mutate(selected.id); }}
          >
            Download UI graph for ComfyUI
          </button>
          {comfyTarget && (
            <a className="secondary compact-button" href={comfyTarget.url} target="_blank" rel="noopener noreferrer">
              Open ComfyUI
            </a>
          )}
        </div>
      )}
      {selectedFamily && <WorkflowFamilyPreferences family={selectedFamily} />}
      <div className="workflow-layout">
        <div className="workflow-list">{workflows.data?.map((workflow) => <button key={workflow.id} className={selected?.id === workflow.id ? "selected" : ""} onClick={() => { setSelectedId(workflow.id); setSelectedRevisionId(workflow.current_revision_id); }}><span><strong>{workflow.name}</strong><small>{workflow.operation} · {workflow.revisions.length} revision{workflow.revisions.length === 1 ? "" : "s"}</small></span></button>)}</div>
        <div className="workflow-detail">{selected && selectedRevision ? <><div className="detail-title"><div><small>{selected.operation}</small><h2>{selected.name}</h2><p>{selected.description}</p></div><div className="row-actions"><button className="secondary compact-button" onClick={openEdit}>New revision</button><button className="secondary compact-button" onClick={() => clone.mutate(selected.id)}>Duplicate</button><button className="secondary compact-button" onClick={() => exportBundle.mutate(selected.id)}>Export</button><button className="secondary compact-button" onClick={() => validate.mutate(selected.id)}>Validate</button></div></div><div className="workflow-revision-bar"><label>Revision<select value={selectedRevision.id} onChange={(event) => setSelectedRevisionId(event.target.value)}>{[...selected.revisions].sort((a, b) => b.version - a.version).map((revision) => <option key={revision.id} value={revision.id}>v{revision.version}{revision.id === selected.current_revision_id ? " · current" : ""}</option>)}</select></label>{selectedRevision.id !== selected.current_revision_id && <button className="secondary compact-button" onClick={() => restore.mutate({ id: selected.id, revisionId: selectedRevision.id })}>Restore as new revision</button>}<span className={`badge ${selectedRevision.trusted ? "likely" : "advanced_import"}`}>{selectedRevision.trusted ? "Trusted" : "Untrusted"}</span></div><section className="workflow-input-section"><h3>Declared controls</h3><WorkflowControls schema={selectedRevision.input_schema_json} /></section><details open><summary>Executable graph</summary><pre>{JSON.stringify(selectedRevision.api_graph_json, null, 2)}</pre></details><details><summary>Dependencies</summary><pre>{JSON.stringify(selectedRevision.dependencies_json, null, 2)}</pre></details>{currentRevision && currentRevision.id !== selectedRevision.id && <details><summary>Compare with current revision</summary><div className="workflow-compare"><pre>{JSON.stringify(selectedRevision.api_graph_json, null, 2)}</pre><pre>{JSON.stringify(currentRevision.api_graph_json, null, 2)}</pre></div></details>}{verdict && <div className={`callout ${verdict.valid ? "success" : "error"}`} role={verdict.valid ? "status" : "alert"}>{verdict.valid ? "Workflow and declared dependencies are valid for the active media engine." : verdict.errors.join("\n")}{verdict.warnings.map((warning) => `\nWarning: ${warning}`)}</div>}</> : <EmptyState icon={<WorkflowIcon />} title="Select a workflow" body="Review its revision, inputs, dependencies, and validation." />}</div>
      </div>
      <RegistryInstallsPanel />
      <CustomNodesPanel />
      {newOpen && (
        <AccessibleDialog
          title={editing ? "Create workflow revision" : "Create ComfyUI workflow"}
          eyebrow={editing ? "Immutable revision" : "Portable workflow"}
          closeLabel="Close workflow editor"
          onClose={() => setNewOpen(false)}
          className="workflow-editor"
        >
          <label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <label>Description<textarea rows={2} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
          <label>Operation<select value={operation} disabled={editing} onChange={(event) => setOperation(event.target.value)}><option value="text_to_image">Text to image</option><option value="image_to_image">Image to image</option><option value="text_to_video">Text to video</option><option value="image_to_video">Image to video</option></select></label>
          <label>API-format workflow JSON<textarea rows={10} value={graph} onChange={(event) => setGraph(event.target.value)} /></label>
          <label>UI workflow JSON<textarea rows={5} value={uiGraph} onChange={(event) => setUiGraph(event.target.value)} /></label>
          <label>Declared input schema JSON<textarea rows={6} value={inputSchema} onChange={(event) => setInputSchema(event.target.value)} /></label>
          <label>Dependencies JSON<textarea rows={5} value={dependencies} onChange={(event) => setDependencies(event.target.value)} /></label>
          <label className="toggle-row"><span><strong>Trust this workflow</strong><small>Only enable after reviewing every node and dependency.</small></span><input type="checkbox" checked={trusted} onChange={(event) => setTrusted(event.target.checked)} /></label>
          {save.error && <ErrorCallout message={save.error.message} />}
          <footer><button className="secondary" onClick={() => setNewOpen(false)}>Cancel</button><button className="primary" disabled={!name.trim() || save.isPending} onClick={() => save.mutate()}>{save.isPending ? "Saving…" : editing ? "Create revision" : "Save workflow"}</button></footer>
        </AccessibleDialog>
      )}
    </div>
  );
}
