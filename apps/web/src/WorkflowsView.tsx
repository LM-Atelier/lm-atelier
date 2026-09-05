import { useEffect, useRef, useState } from "react";
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
import {
  openWorkflowEditorPopup,
  runWorkflowEditor,
  type WorkflowEditorPhase,
  type WorkflowEditorSubmission,
} from "./workflowEditorBridge";
import type { WorkflowEditorReturn } from "./types";

const editorPhaseLabel: Record<WorkflowEditorPhase, string> = {
  preparing: "Preparing the native editor…",
  connecting: "Connecting to ComfyUI…",
  loading: "Loading the workflow…",
  editing: "Editing in ComfyUI…",
  validating: "Validating the returned workflow…",
  saving: "Saving a review draft…",
};

/** What a revision declares it can be asked for, as a description of it.
 *
 * These used to be live inputs backed by local state, in a pane whose whole
 * job is to show what a workflow is. Nothing consumed them: typing changed a
 * value that was read by nobody, saved by nothing, and discarded when the
 * selection moved. An editable-looking field that throws away what you put in
 * it is worse than plain text, because it invites the attempt.
 *
 * Where these controls are really answered - a turn in chat, an apply in the
 * studio - the settings panel there owns them, validates them, and sends
 * them.
 */
export function WorkflowControls({ schema }: { schema: Record<string, unknown> }) {
  const properties = schema.properties && typeof schema.properties === "object"
    ? schema.properties as Record<string, Record<string, unknown>>
    : {};
  const required = Array.isArray(schema.required) ? schema.required.map(String) : [];
  if (!Object.keys(properties).length) {
    return <p className="muted">This revision does not declare user-facing inputs.</p>;
  }
  return (
    <dl className="workflow-controls">
      {Object.entries(properties).map(([key, field]) => {
        const label = String(field.title ?? key.replaceAll("_", " "));
        const description = typeof field.description === "string" ? field.description : "";
        const choices = Array.isArray(field.enum) ? field.enum : [];
        const bounds = [
          typeof field.minimum === "number" ? `min ${field.minimum}` : null,
          typeof field.maximum === "number" ? `max ${field.maximum}` : null,
        ].filter(Boolean).join(", ");
        return (
          <div key={key} className="workflow-control">
            <dt>
              <strong>{label}</strong>
              <small>{String(field.type ?? "string")}</small>
              {required.includes(key) && <small className="badge">required</small>}
            </dt>
            <dd>
              {description && <p>{description}</p>}
              {field.default !== undefined && <p><small>Default: {String(field.default)}</small></p>}
              {choices.length > 0 && (
                <p><small>One of: {choices.map((choice) => String(choice)).join(", ")}</small></p>
              )}
              {bounds && <p><small>{bounds}</small></p>}
            </dd>
          </div>
        );
      })}
    </dl>
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
  const importInput = useRef<HTMLInputElement>(null);
  const editorAbort = useRef<AbortController | null>(null);
  const editorStarting = useRef(false);
  const [editorPhase, setEditorPhase] = useState<WorkflowEditorPhase | null>(null);
  const [editorNotice, setEditorNotice] = useState<string | null>(null);
  const [popupError, setPopupError] = useState<Error | null>(null);
  const [pendingDraft, setPendingDraft] = useState<{
    workflowId: string;
    returned: WorkflowEditorReturn;
  } | null>(null);
  const [pendingSubmission, setPendingSubmission] = useState<WorkflowEditorSubmission | null>(null);
  useEffect(() => () => editorAbort.current?.abort(), []);
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
      const revision = { engine_version: null, api_graph: JSON.parse(graph), ui_graph: JSON.parse(uiGraph), input_schema: JSON.parse(inputSchema), dependencies: JSON.parse(dependencies) };
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
  const [comfyTarget, setComfyTarget] = useState<{ url: string; filename: string } | null>(null);
  const downloadForComfy = useMutation({
    mutationFn: (id: string) => api.workflowOpenTarget(id),
    onSuccess: (target) => {
      setComfyTarget({ url: target.url, filename: target.filename });
      downloadJson(target.ui_graph, target.filename);
    },
  });
  const nativeEditor = useMutation({
    mutationFn: ({
      id,
      popup,
      controller,
    }: {
      id: string;
      popup: Window;
      controller: AbortController;
    }) => runWorkflowEditor(api, id, popup, {
      signal: controller.signal,
      onPhase: setEditorPhase,
      onValidated: (returned) => setPendingDraft(
        returned?.changed ? { workflowId: id, returned } : null,
      ),
      onSubmission: setPendingSubmission,
      onExpiryWarning: () => setEditorNotice(
        "The secure editor session expires in five minutes. Save or download the workflow in ComfyUI.",
      ),
    }),
    onSuccess: (result, variables) => {
      setPendingDraft(null);
      setSelectedId(variables.id);
      if (result.kind === "draft") {
        setSelectedRevisionId(result.draft.draft_revision_id);
        setEditorNotice(`Draft v${result.draft.version} saved for review.`);
      } else {
        setEditorNotice("No workflow changes to save.");
      }
      refresh();
    },
    onSettled: (_result, _error, variables) => {
      if (editorAbort.current === variables.controller) {
        editorAbort.current = null;
        editorStarting.current = false;
      }
      setEditorPhase(null);
    },
  });
  const retryDraft = useMutation({
    mutationFn: ({ workflowId, returned }: NonNullable<typeof pendingDraft>) =>
      api.createWorkflowEditorDraft(workflowId, returned.validated_return_id),
    onMutate: () => nativeEditor.reset(),
    onSuccess: (draft, variables) => {
      setPendingDraft(null);
      setSelectedId(variables.workflowId);
      setSelectedRevisionId(draft.draft_revision_id);
      setEditorNotice(`Draft v${draft.version} saved for review.`);
      refresh();
    },
  });
  const retrySubmission = useMutation({
    mutationFn: async (submission: WorkflowEditorSubmission) => {
      const returned = await api.consumeWorkflowEditor(
        submission.workflowId,
        submission.sessionId,
        {
          nonce: submission.nonce,
          base_revision_id: submission.baseRevisionId,
          ui_graph: submission.uiGraph,
          api_prompt: submission.apiPrompt,
        },
      );
      return { submission, returned };
    },
    onMutate: () => nativeEditor.reset(),
    onSuccess: ({ submission, returned }) => {
      setPendingSubmission(null);
      if (!returned.changed) {
        setEditorNotice("No workflow changes to save.");
        return;
      }
      const pending = { workflowId: submission.workflowId, returned };
      setPendingDraft(pending);
      retryDraft.mutate(pending);
    },
  });
  const openNativeEditor = () => {
    if (
      !selected
      || editorStarting.current
      || pendingSubmission
      || pendingDraft
      || retryDraft.isPending
      || retrySubmission.isPending
    ) return;
    editorStarting.current = true;
    nativeEditor.reset();
    retrySubmission.reset();
    retryDraft.reset();
    setPopupError(null);
    setEditorNotice(null);
    const popup = openWorkflowEditorPopup();
    if (!popup) {
      editorStarting.current = false;
      setPopupError(new Error("The browser blocked the workflow editor window. Allow popups for this local app and try again."));
      return;
    }
    const controller = new AbortController();
    editorAbort.current = controller;
    nativeEditor.mutate({ id: selected.id, popup, controller });
  };
  const discardPendingEdit = () => {
    if (pendingSubmission) {
      void api.cancelWorkflowEditor(
        pendingSubmission.workflowId,
        pendingSubmission.sessionId,
        pendingSubmission.nonce,
      ).catch(() => {});
    }
    setPendingSubmission(null);
    setPendingDraft(null);
    nativeEditor.reset();
    retrySubmission.reset();
    retryDraft.reset();
    setEditorNotice("Pending workflow edit discarded.");
  };
  const {
    importFile: importBundle,
    importError,
    packageReview,
    closePackageReview,
  } = useWorkflowPackageImport(refresh);
  const openCreate = () => { setEditing(false); setName("Custom image workflow"); setDescription(""); setOperation("text_to_image"); setGraph("{}"); setUiGraph("{}"); setInputSchema("{}"); setDependencies("{}"); setNewOpen(true); };
  const openEdit = () => { if (!selected) return; const revision = selected.revisions.find((item) => item.id === selected.current_revision_id) ?? selected.revisions.at(-1); if (!revision) return; setEditing(true); setName(selected.name); setDescription(selected.description); setOperation(selected.operation); setGraph(JSON.stringify(revision.api_graph_json, null, 2)); setUiGraph(JSON.stringify(revision.ui_graph_json, null, 2)); setInputSchema(JSON.stringify(revision.input_schema_json, null, 2)); setDependencies(JSON.stringify(revision.dependencies_json, null, 2)); setNewOpen(true); };
  // A verdict is about the workflow that was validated. Held globally by
  // the mutation it stayed on screen when the selection moved, reading as
  // the new workflow's result - the right answer under the wrong name.
  const verdict = validate.data && validate.variables === selected?.id ? validate.data : null;
  const selectedRevision = selected?.revisions.find((revision) => revision.id === selectedRevisionId) ?? selected?.revisions.find((revision) => revision.id === selected.current_revision_id) ?? selected?.revisions.at(-1);
  const currentRevision = selected?.revisions.find((revision) => revision.id === selected.current_revision_id);
  const editorRecoveryReady = !nativeEditor.isPending
    && !retrySubmission.isPending
    && !retryDraft.isPending;
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
      {(importError || clone.error || restore.error || exportBundle.error || downloadForComfy.error || nativeEditor.error || retrySubmission.error || retryDraft.error || popupError || validate.error) && <ErrorCallout message={(importError || clone.error || restore.error || exportBundle.error || downloadForComfy.error || nativeEditor.error || retrySubmission.error || retryDraft.error || popupError || validate.error)?.message} />}
      {packageReview && <WorkflowPackageReview analysis={packageReview.analysis} fileName={packageReview.fileName} uiGraph={packageReview.uiGraph} onImported={() => { closePackageReview(); refresh(); }} onClose={closePackageReview} />}
      {selected && (
        <div className="storage-actions">
          <button
            className="primary"
            disabled={nativeEditor.isPending || Boolean(pendingSubmission || pendingDraft) || selectedRevision?.id !== selected.current_revision_id}
            onClick={openNativeEditor}
          >
            {nativeEditor.isPending
              ? "Editor open"
              : selectedRevision?.id !== selected.current_revision_id
                ? "Select current revision to edit"
                : "Edit in ComfyUI (preview)"}
          </button>
          <button
            className="secondary compact-button"
            disabled={downloadForComfy.isPending}
            onClick={() => { setComfyTarget(null); downloadForComfy.mutate(selected.id); }}
          >
            Download UI graph
          </button>
          {comfyTarget && (
            <a className="secondary compact-button" href={comfyTarget.url} target="_blank" rel="noopener noreferrer">
              Open ComfyUI manually
            </a>
          )}
          {editorPhase && <span role="status" className="muted">{editorPhaseLabel[editorPhase]}</span>}
          {pendingDraft && editorRecoveryReady && (
            <button
              className="secondary compact-button"
              disabled={retryDraft.isPending}
              onClick={() => retryDraft.mutate(pendingDraft)}
            >
              {retryDraft.isPending ? "Saving…" : "Retry saving validated edit"}
            </button>
          )}
          {pendingSubmission && editorRecoveryReady && (
            <button
              className="secondary compact-button"
              disabled={retrySubmission.isPending}
              onClick={() => retrySubmission.mutate(pendingSubmission)}
            >
              {retrySubmission.isPending ? "Validating…" : "Retry validating returned edit"}
            </button>
          )}
          {(pendingSubmission || pendingDraft) && editorRecoveryReady && (
            <button
              className="secondary compact-button danger"
              disabled={retrySubmission.isPending || retryDraft.isPending}
              onClick={discardPendingEdit}
            >
              Discard pending edit
            </button>
          )}
          {editorNotice && <span role="status" className="muted">{editorNotice}</span>}
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
          {save.error && <ErrorCallout message={save.error.message} />}
          <footer><button className="secondary" onClick={() => setNewOpen(false)}>Cancel</button><button className="primary" disabled={!name.trim() || save.isPending} onClick={() => save.mutate()}>{save.isPending ? "Saving…" : editing ? "Create revision" : "Save workflow"}</button></footer>
        </AccessibleDialog>
      )}
    </div>
  );
}
