import { useState } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import { WorkflowSelector } from "./WorkflowSelector";
import { useConfirm } from "./useConfirm";
import type { EngineCapabilities, EngineRole, GenerationPreset, Project } from "./types";

/** Everything a project can be told about itself.
 *
 * Moved out of App.tsx as its own change. It depended on the generation
 * settings panel, which is why it could not come out until that did.
 */

export function ProjectManager({
  project,
  engines,
  presets,
  onClose,
  onSave,
  onDelete,
  onExport,
}: {
  project: Project;
  engines: EngineCapabilities[];
  presets: GenerationPreset[];
  onClose: () => void;
  onSave: (values: Partial<Project>) => void;
  onDelete: () => void;
  onExport: (includeMedia: boolean) => void;
}) {
  const [confirmDialog, confirm] = useConfirm();
  const [name, setName] = useState(project.name);
  const [description, setDescription] = useState(project.description);
  const [instructions, setInstructions] = useState(project.instructions);
  const [archived, setArchived] = useState(project.archived);
  const [settingsRole, setSettingsRole] = useState<EngineRole>("chat");
  const [generationSettings, setGenerationSettings] = useState<
    NonNullable<Project["generation_settings_json"]>
  >(() => Object.fromEntries(
    Object.entries(project.generation_settings_json ?? {}).map(([role, settings]) => [
      role,
      { ...settings },
    ]),
  ));
  const [generationPresetIds, setGenerationPresetIds] = useState<
    NonNullable<Project["generation_preset_ids_json"]>
  >({ ...(project.generation_preset_ids_json ?? {}) });
  const setRoleSettings = (values: Record<string, unknown>) => {
    setGenerationSettings((current) => {
      const next = { ...current };
      if (Object.keys(values).length) next[settingsRole] = values;
      else delete next[settingsRole];
      return next;
    });
  };
  const setRolePreset = (presetId: string | null) => {
    setGenerationPresetIds((current) => {
      const next = { ...current };
      if (presetId) next[settingsRole] = presetId;
      else delete next[settingsRole];
      return next;
    });
  };
  const clearRoleDefaults = () => {
    setRoleSettings({});
    setRolePreset(null);
  };
  const clearAllDefaults = () => {
    setGenerationSettings({});
    setGenerationPresetIds({});
  };
  return (
    <AccessibleDialog
      title="Manage project"
      eyebrow="Workspace"
      closeLabel="Close project manager"
      onClose={onClose}
      className="workspace-editor project-editor"
    >
      <label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label>Description<textarea rows={3} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
      <label>Project instructions<textarea rows={5} value={instructions} onChange={(event) => setInstructions(event.target.value)} /></label>
      <section className="project-generation-defaults" aria-labelledby="project-generation-defaults-heading">
        <div className="project-defaults-heading">
          <div>
            <h3 id="project-generation-defaults-heading">Generation defaults</h3>
            <p>Chats inherit these settings; chat and turn choices override them.</p>
          </div>
          <button className="secondary compact-button" type="button" onClick={clearAllDefaults}>Clear all</button>
        </div>
        <div className="segmented project-role-tabs" aria-label="Project generation role">
          {(["chat", "image", "video"] as EngineRole[]).map((role) => (
            <button
              key={role}
              type="button"
              className={settingsRole === role ? "active" : ""}
              aria-pressed={settingsRole === role}
              onClick={() => setSettingsRole(role)}
            >
              {role}
            </button>
          ))}
        </div>
        <GenerationSettingsPanel
          key={settingsRole}
          role={settingsRole}
          engines={engines}
          values={generationSettings[settingsRole] ?? {}}
          onValues={setRoleSettings}
          presets={presets}
          presetId={generationPresetIds[settingsRole] ?? null}
          onPreset={setRolePreset}
          presetLabel={`${settingsRole} project preset`}
          resetLabel="Clear role defaults"
          onReset={clearRoleDefaults}
        />
      </section>
      {/* A project's choice is what its chats fall back to, which is why
          "inherit" and "use the project's choice" are the same idea seen
          from two levels. */}
      <section className="workflow-selectors">
        <h3>Workflows for this project</h3>
        <div>
          <WorkflowSelector scope="project" scopeId={project.id} capability="image" label="Images" />
          <WorkflowSelector scope="project" scopeId={project.id} capability="video" label="Video" />
        </div>
      </section>
      <label className="toggle-row"><span><strong>Archived</strong><small>Hide this project while preserving its chats and media.</small></span><input type="checkbox" checked={archived} onChange={(event) => setArchived(event.target.checked)} /></label>
      <div className="project-export-actions"><button className="secondary" onClick={() => onExport(false)}>Export metadata only</button><button className="secondary" onClick={() => onExport(true)}>Export with media</button></div>
      <footer className="editor-actions">
        <button className="secondary danger" onClick={() => void confirm({ title: `Delete ${project.name}?`, question: "The chats inside it are kept, but become unfiled.", confirmLabel: "Delete project" }).then((ok) => ok && onDelete())}>Delete project</button>
        <button className="secondary" onClick={onClose}>Cancel</button>
        <button
          className="primary"
          disabled={!name.trim()}
          onClick={() => onSave({
            name: name.trim(),
            description,
            instructions,
            archived,
            generation_settings_json: generationSettings,
            generation_preset_ids_json: generationPresetIds,
          })}
        >
          Save project
        </button>
      </footer>
      {confirmDialog}
    </AccessibleDialog>
  );
}
