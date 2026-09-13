import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";

function count(value: number, one: string, many: string): string {
  return `${value.toLocaleString()} ${value === 1 ? one : many}`;
}

/** How much space the workspace is using, and how much of it could be given back.
 *
 * Read-only on purpose: every figure here comes from the server's own
 * accounting, and the places that act on it - the Model Library's partial
 * download cleanup, retention - already exist. Showing them together is what
 * was missing, so somebody short of space can see where it went.
 *
 * Mounted only while Data & backups is open. Working out what retention could
 * clear means examining every stored file, which is not a cost to pay while
 * nobody is looking.
 */
export function StorageSummary() {
  const media = useQuery({ queryKey: ["artifact-storage"], queryFn: api.artifactStorage });
  const models = useQuery({ queryKey: ["model-storage"], queryFn: api.modelStorage });

  return (
    <section aria-labelledby="storage-summary-heading">
      <div className="detail-title">
        <div><h2 id="storage-summary-heading">Storage</h2><p>What the workspace keeps on this computer.</p></div>
      </div>
      {media.data?.warning && (
        <div className="callout error" role="alert">
          Free disk space is low: {formatBytes(media.data.disk_free_bytes)} left.
        </div>
      )}
      <dl className="storage-summary">
        {models.data && (
          <>
            <div>
              <dt>Models</dt>
              <dd>
                {count(models.data.installed_count, "installed model", "installed models")} ·{" "}
                {formatBytes(models.data.installed_bytes)}
              </dd>
            </div>
            {models.data.partial_download_count > 0 && (
              <div>
                <dt>Unfinished downloads</dt>
                <dd>
                  {count(models.data.partial_download_count, "download", "downloads")} ·{" "}
                  {formatBytes(models.data.partial_download_bytes)}
                </dd>
              </div>
            )}
            <div>
              <dt>Catalog cache</dt>
              <dd>{formatBytes(models.data.catalog_cache_bytes)}</dd>
            </div>
          </>
        )}
        {media.data && (
          <>
            <div>
              <dt>Images and videos</dt>
              <dd>
                {count(media.data.total_count, "file", "files")} · {formatBytes(media.data.total_bytes)}
              </dd>
            </div>
            <div>
              <dt>In use</dt>
              <dd>
                {count(media.data.referenced_count, "file", "files")} · {formatBytes(media.data.referenced_bytes)}
                <small>kept by a chat, the Media Library, a reference or a past run</small>
              </dd>
            </div>
            <div>
              <dt>Previews and in-between steps</dt>
              <dd>
                {count(media.data.temporary_count, "file", "files")} · {formatBytes(media.data.temporary_bytes)}
                <small>cleared once nothing uses them and they are {count(media.data.temporary_retention_hours, "hour", "hours")} old</small>
              </dd>
            </div>
            <div>
              <dt>Ready to clear</dt>
              <dd>
                {count(media.data.eligible_count, "file", "files")} · {formatBytes(media.data.eligible_bytes)}
                <small>unused, unfavorited and past their time: {count(media.data.retention_days, "day", "days")} unused, or the preview limit</small>
              </dd>
            </div>
            {Boolean(media.data.retention_pending_count) && (
              <div>
                <dt>Waiting to be cleared</dt>
                <dd>
                  {count(media.data.retention_pending_count ?? 0, "file", "files")}
                  <small>unused, but for less than {count(media.data.retention_days, "day", "days")} so far</small>
                </dd>
              </div>
            )}
            <div>
              <dt>Free disk space</dt>
              <dd>{formatBytes(media.data.disk_free_bytes)}</dd>
            </div>
          </>
        )}
      </dl>
      {(media.error || models.error) && <ErrorCallout message="Storage figures are unavailable right now." />}
    </section>
  );
}
