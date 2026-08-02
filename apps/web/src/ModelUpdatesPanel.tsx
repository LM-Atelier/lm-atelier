import { useMutation } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import type { CatalogModel, ModelUpdate } from "./types";

/** On-demand staleness check for installed assets with exact provider versions.
 *
 * Nothing polls: the button is the only thing that asks the provider, and the
 * report separates "update available", "up to date", and "could not check"
 * instead of collapsing uncertainty into either verdict. Installing an update
 * hands the newer version's catalog card to the normal verified install flow -
 * full preflight, hashes, and confirmation, nothing abbreviated.
 */
export function ModelUpdatesPanel({
  onInstall,
}: {
  onInstall: (model: CatalogModel, selectedRole: string) => void;
}) {
  const check = useMutation({ mutationFn: () => api.modelUpdates() });
  const review = useMutation({
    mutationFn: async (update: ModelUpdate) => {
      const role = update.kind === "lora" ? "lora" : null;
      const detail = await api.catalogItemDetail("civitai", update.update_version_id ?? "", role);
      return { model: detail.model, role: role ?? "image" };
    },
    onSuccess: ({ model, role }) => onInstall(model, role),
  });
  const report = check.data;
  const updates = report?.filter((update) => update.state === "update_available") ?? [];
  const current = report?.filter((update) => update.state === "current") ?? [];
  const unknown = report?.filter((update) => update.state === "unknown") ?? [];
  const error = check.error ?? review.error;
  return (
    <section className="model-updates">
      <div className="storage-actions">
        <button
          className="secondary compact-button"
          disabled={check.isPending}
          onClick={() => check.mutate()}
        >
          <RefreshCw size={16} />
          {check.isPending ? "Checking versions..." : "Check for updates"}
        </button>
        {report && (
          <span className="storage-pill">
            {updates.length === 0
              ? "Everything checkable is up to date"
              : `${updates.length} update${updates.length === 1 ? "" : "s"} available`}
            {` · ${current.length} current`}
            {unknown.length > 0 ? ` · ${unknown.length} unreachable` : ""}
          </span>
        )}
      </div>
      {error && <ErrorCallout message={(error as Error).message} />}
      {report && report.length === 0 && (
        <p className="package-review-note">
          Nothing installed names an exact provider version yet, so there is nothing to compare.
        </p>
      )}
      {updates.length > 0 && (
        <div className="profile-table">
          {updates.map((update) => (
            <div key={update.install_id}>
              <span className="badge advanced_import">Update</span>
              <span>
                <strong>{update.name}</strong>
                <small>
                  {update.installed_version_name ?? update.installed_version_id}
                  {" -> "}
                  {update.update_version_name ?? update.update_version_id}
                  {update.update_published_at ? ` · ${update.update_published_at.slice(0, 10)}` : ""}
                  {update.update_base_model ? ` · ${update.update_base_model}` : ""}
                </small>
              </span>
              {update.update_changelog && (
                <details>
                  <summary>What changed</summary>
                  <p>{update.update_changelog}</p>
                </details>
              )}
              <span className="row-actions">
                {update.kind === "lora" ? (
                  <button
                    className="secondary compact-button"
                    disabled={review.isPending}
                    onClick={() => review.mutate(update)}
                  >
                    Review update
                  </button>
                ) : (
                  <small>Install the new version from the catalog to update.</small>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
      {unknown.length > 0 && (
        <p className="package-review-note">
          Could not check: {unknown.map((update) => update.name).join(", ")}.
        </p>
      )}
    </section>
  );
}
