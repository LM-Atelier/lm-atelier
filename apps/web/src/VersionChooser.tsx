import { useQuery } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes, formatDate } from "./format";
import { api } from "./api";
import type { CatalogVersionRow } from "./types";

/** Pick which version of a model to install.
 *
 * A grouped card names the model; this is where the version is chosen. That
 * choice is never made for you: the card opens this rather than installing,
 * because a version is what actually lands on disk and what the download path
 * is bound to.
 *
 * The versions endpoint is CivitAI's, so this is only offered for CivitAI
 * cards. A second provider with a parent identity would need its own route
 * before its cards could open this, and offering it early would send a model
 * id to a source that has never heard of it.
 *
 * Installed state has three answers. Where a kind records no provider version
 * - checkpoints today - the row says so plainly instead of claiming the
 * version is absent, since "not installed" about something we cannot see is
 * how a person ends up with a second copy.
 */
export function VersionChooser({
  modelId,
  modelName,
  onChoose,
  onClose,
}: {
  modelId: string;
  modelName: string;
  onChoose: (versionId: string) => void;
  onClose: () => void;
}) {
  const versions = useQuery({
    queryKey: ["catalog-versions", modelId],
    queryFn: () => api.catalogVersions(modelId),
  });

  return (
    <AccessibleDialog
      title={modelName}
      eyebrow="Choose a version"
      closeLabel="Close version chooser"
      onClose={onClose}
      className="version-chooser"
    >
      {versions.isPending && <p>Reading the versions…</p>}
      {versions.error && <ErrorCallout message={(versions.error as Error).message} />}
      <ul className="version-list">
        {(versions.data?.versions ?? []).map((row) => (
          <li key={row.version_id}>
            <div>
              <strong>{row.version_name || row.version_id}</strong>
              <small>
                {row.base_model ? `${row.base_model} · ` : ""}
                {row.size_bytes ? `${formatBytes(row.size_bytes)} · ` : ""}
                {formatDate(row.published_at ?? null)}
              </small>
              {row.changelog && <p className="version-changelog">{row.changelog}</p>}
            </div>
            {installedLabel(row)}
            <button
              className="secondary compact-button"
              disabled={row.installed === true}
              onClick={() => onChoose(row.version_id)}
            >
              {row.installed === true ? "Installed" : "Install this version"}
            </button>
          </li>
        ))}
      </ul>
      {versions.data?.versions.length === 0 && <p>This model lists no installable versions.</p>}
    </AccessibleDialog>
  );
}

function installedLabel(row: CatalogVersionRow) {
  if (row.installed === true) {
    return <span className="badge likely">{row.installed_as ?? "Installed"}</span>;
  }
  if (row.installed === false) return <span className="badge">Not installed</span>;
  // Unknown is a real answer, and saying nothing is how it is told.
  return null;
}
