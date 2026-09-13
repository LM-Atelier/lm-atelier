import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef } from "react";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { useProjectMutations } from "./useProjectMutations";

/** Taking a project out of the workspace as an archive, and bringing one back in.
 *
 * Both already existed, one in each project's own menu and one beside the
 * project list, which is where somebody looking to move their work to another
 * computer is least likely to look. Data & backups is where they look, so the
 * same actions are offered here too, through the same mutations, so an archive
 * made here is exactly the archive made from the project.
 *
 * Archived projects are listed as well: an archived project is still somebody's
 * work, and exporting it is one of the few things left to do with it.
 */
export function ProjectArchives() {
  const client = useQueryClient();
  // Importing from the sidebar opens the imported chat; from Settings the
  // person stays where they are and is told what arrived.
  const { exportProject, importProject } = useProjectMutations({ client, onImportedChat: () => undefined });
  const projects = useQuery({ queryKey: ["projects", "including-archived"], queryFn: () => api.projects(true) });
  const picker = useRef<HTMLInputElement>(null);
  const error = exportProject.error ?? importProject.error ?? projects.error;

  return (
    <section aria-labelledby="project-archives-heading">
      <div className="detail-title storage-actions">
        <div>
          <h2 id="project-archives-heading">Project archives</h2>
          <p>Move a project, its chats and optionally its media, to another workspace.</p>
        </div>
        <div className="row-actions storage-actions">
          <input
            ref={picker}
            hidden
            type="file"
            aria-label="Project archive to import"
            accept=".zip,.lm-atelier.zip,application/zip"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) importProject.mutate(file);
            }}
          />
          <button
            type="button"
            className="secondary"
            aria-disabled={importProject.isPending}
            onClick={() => {
              if (!importProject.isPending) picker.current?.click();
            }}
          >
            {importProject.isPending ? "Importing…" : "Import an archive"}
          </button>
        </div>
      </div>
      {importProject.data && (
        <div className="callout success" role="status">Imported {importProject.data.name}.</div>
      )}
      {projects.data && projects.data.length === 0 && <p className="muted">No projects to export yet.</p>}
      {projects.data && projects.data.length > 0 && (
        <div className="backup-list">
          {projects.data.map((project) => (
            <div className="backup-row" key={project.id}>
              <span className="backup-copy">
                <strong>{project.name}</strong>
                {project.archived && <small>Archived</small>}
              </span>
              <span className="row-actions">
                <button
                  type="button"
                  className="secondary compact-button"
                  aria-label={`Export ${project.name}, metadata only`}
                  onClick={() => exportProject.mutate({ id: project.id, includeMedia: false })}
                >
                  Metadata only
                </button>
                <button
                  type="button"
                  className="secondary compact-button"
                  aria-label={`Export ${project.name} with media`}
                  onClick={() => exportProject.mutate({ id: project.id, includeMedia: true })}
                >
                  With media
                </button>
              </span>
            </div>
          ))}
        </div>
      )}
      {error && <ErrorCallout message={error.message} />}
    </section>
  );
}
