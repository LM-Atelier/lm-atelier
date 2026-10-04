import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { useProjectPages } from "./useProjectPages";
import { ProjectPageControls } from "./ProjectPageControls";
import { ErrorCallout } from "./ErrorCallout";
import { isEncryptedArchive } from "./projectArchiveFiles";
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
 *
 * An export can be encrypted with a passphrase, typed twice so a slip of the
 * keyboard does not lock the archive for good. An encrypted archive chosen for
 * import is recognized by its first bytes, and its passphrase is asked for
 * before anything is sent. Neither passphrase is kept anywhere but this page.
 */
export function ProjectArchives() {
  const client = useQueryClient();
  // Importing from the sidebar opens the imported chat; from Settings the
  // person stays where they are and is told what arrived.
  const { exportProject, importProject } = useProjectMutations({ client });
  const [search, setSearch] = useState("");
  const projects = useProjectPages(search, true);
  const picker = useRef<HTMLInputElement>(null);
  const [encrypt, setEncrypt] = useState(false);
  const [passphrase, setPassphrase] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [locked, setLocked] = useState<File | null>(null);
  const [unlock, setUnlock] = useState("");
  const unlockField = useRef<HTMLInputElement>(null);
  const hint = useId();
  // While the passphrase prompt is open, an import's error belongs inside it.
  const error = locked ? exportProject.error : (exportProject.error ?? importProject.error);
  const ready = !encrypt || (passphrase.length > 0 && passphrase === confirmation);
  useEffect(() => {
    if (locked) unlockField.current?.focus();
  }, [locked]);
  const exportWith = (id: string, includeMedia: boolean) => {
    if (!ready) return;
    exportProject.mutate(encrypt ? { id, includeMedia, passphrase } : { id, includeMedia });
  };
  const closeLocked = () => {
    setLocked(null);
    setUnlock("");
    importProject.reset();
  };
  const choose = async (file: File) => {
    importProject.reset();
    if (await isEncryptedArchive(file).catch(() => false)) {
      setLocked(file);
      setUnlock("");
      return;
    }
    setLocked(null);
    setUnlock("");
    importProject.mutate(file);
  };
  const openLocked = () => {
    if (!locked || !unlock || importProject.isPending) return;
    importProject.mutate({ file: locked, passphrase: unlock }, { onSuccess: () => { setLocked(null); setUnlock(""); } });
  };

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
            accept=".zip,.lm-atelier.zip,.lm-atelier.encrypted,application/zip,application/octet-stream"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) void choose(file);
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
      {locked && (
        <form
          className="callout"
          aria-label="Encrypted archive"
          onSubmit={(event) => { event.preventDefault(); openLocked(); }}
        >
          <p>{locked.name} is encrypted. Enter its passphrase to import it.</p>
          <label>Archive passphrase<input ref={unlockField} type="password" autoComplete="current-password" value={unlock} onChange={(event) => setUnlock(event.target.value)} /></label>
          {importProject.error && <ErrorCallout message={importProject.error.message} />}
          <span className="row-actions">
            <button type="submit" aria-disabled={!unlock || importProject.isPending}>Import</button>
            <button type="button" className="secondary" aria-disabled={importProject.isPending}
              onClick={() => { if (!importProject.isPending) closeLocked(); }}>Cancel</button>
          </span>
        </form>
      )}
      {importProject.data && (
        <div className="callout success" role="status">Imported {importProject.data.name}.</div>
      )}
      <label className="toggle-row">
        <span><strong>Encrypt exports with a passphrase</strong><small>The archive cannot be opened without it, and a forgotten passphrase cannot be recovered.</small></span>
        <input type="checkbox" checked={encrypt} onChange={(event) => setEncrypt(event.target.checked)} />
      </label>
      {encrypt && (
        <span className="row-actions">
          <label>Passphrase<input type="password" autoComplete="new-password" maxLength={1024} value={passphrase} onChange={(event) => setPassphrase(event.target.value)} /></label>
          <label>Confirm passphrase<input type="password" autoComplete="new-password" maxLength={1024} value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></label>
        </span>
      )}
      {encrypt && !ready && (
        <p className="muted" id={hint}>
          {confirmation.length > 0 && passphrase !== confirmation
            ? "The passphrases do not match."
            : "Type the passphrase twice to export encrypted."}
        </p>
      )}
      <label>Search projects to export<input type="search" maxLength={500} value={search} onChange={(event) => setSearch(event.target.value)} /></label>
      {projects.data && projects.data.length === 0 && !projects.error && <p className="muted">{search.trim() ? "No matching projects." : "No projects to export yet."}</p>}
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
                  aria-disabled={!ready}
                  aria-describedby={ready ? undefined : hint}
                  onClick={() => exportWith(project.id, false)}
                >
                  Metadata only
                </button>
                <button
                  type="button"
                  className="secondary compact-button"
                  aria-label={`Export ${project.name} with media`}
                  aria-disabled={!ready}
                  aria-describedby={ready ? undefined : hint}
                  onClick={() => exportWith(project.id, true)}
                >
                  With media
                </button>
              </span>
            </div>
          ))}
        </div>
      )}
      {error && <ErrorCallout message={error.message} />}
      <ProjectPageControls pages={projects} />
    </section>
  );
}
