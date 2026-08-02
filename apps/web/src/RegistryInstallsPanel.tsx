import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { activationErrorDescription } from "./registryPreparationErrors";
import type { RegistryInstall } from "./types";

/** Prepared Registry packages and the two decisions each one waits for.
 *
 * Preparation only stages files; nothing here runs until the exact package is
 * explicitly trusted, and nothing loads into ComfyUI until it is separately
 * activated. Revoking trust also deactivates. Every action needs the media
 * worker stopped, and the server refuses with a stable code when it is not.
 */
export function RegistryInstallsPanel() {
  const client = useQueryClient();
  const installs = useQuery({ queryKey: ["registry-installs"], queryFn: api.registryInstalls });
  const refresh = () => void client.invalidateQueries({ queryKey: ["registry-installs"] });
  const review = useMutation({
    mutationFn: ({ id, trusted }: { id: string; trusted: boolean }) =>
      api.reviewRegistryInstall(id, trusted),
    onSuccess: refresh,
  });
  const activate = useMutation({
    mutationFn: (id: string) => api.activateRegistryInstall(id),
    onSuccess: refresh,
  });
  const deactivate = useMutation({
    mutationFn: (id: string) => api.deactivateRegistryInstall(id),
    onSuccess: refresh,
  });
  const error = (review.error ?? activate.error ?? deactivate.error) as
    | (Error & { code?: string })
    | null;
  if (!installs.data?.length) return null;
  return (
    <section className="custom-nodes registry-installs">
      <div className="detail-title">
        <div>
          <h2>Prepared packages</h2>
          <p>
            Prepared code stays inert until you trust the exact package, and inactive until you
            activate it. Stop the media worker before changing either.
          </p>
        </div>
      </div>
      {error && <ErrorCallout message={activationErrorDescription(error)} />}
      <div className="profile-table custom-node-list">
        {installs.data.map((install) => (
          <div key={install.id}>
            <span className={`badge ${install.trusted ? "likely" : "advanced_import"}`}>
              {install.trusted ? "Trusted" : "Review required"}
            </span>
            <span className={`badge ${install.active ? "likely" : "advanced_import"}`}>
              {install.active ? "Active" : "Inactive"}
            </span>
            <span>
              <strong>{install.package_id}</strong>
              <small>
                {install.package_version}
                {` · ${install.node_types.length} node type${install.node_types.length === 1 ? "" : "s"}`}
              </small>
            </span>
            <details>
              <summary>Identity</summary>
              <pre>{identityLines(install)}</pre>
            </details>
            <span className="row-actions">
              <button
                className="secondary compact-button"
                onClick={() =>
                  install.trusted
                    ? review.mutate({ id: install.id, trusted: false })
                    : window.confirm(
                        "I reviewed this exact package and trust its code to run in ComfyUI.",
                      ) && review.mutate({ id: install.id, trusted: true })
                }
              >
                {install.trusted ? "Revoke trust" : "Trust package"}
              </button>
              {install.trusted && !install.active && (
                <button
                  className="secondary compact-button"
                  onClick={() => activate.mutate(install.id)}
                >
                  Activate
                </button>
              )}
              {install.active && (
                <button
                  className="secondary compact-button"
                  onClick={() => deactivate.mutate(install.id)}
                >
                  Deactivate
                </button>
              )}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function identityLines(install: RegistryInstall): string {
  return [
    `archive  ${install.archive_sha256}`,
    `manifest ${install.manifest_sha256}`,
    install.wheel_closure_sha256 ? `closure  ${install.wheel_closure_sha256}` : null,
    install.wheel_environment_sha256 ? `wheels   ${install.wheel_environment_sha256}` : null,
    install.reviewed_at ? `reviewed  ${install.reviewed_at}` : null,
    install.activated_at ? `activated ${install.activated_at}` : null,
  ]
    .filter(Boolean)
    .join("\n");
}
