import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
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
  const [trusting, setTrusting] = useState<RegistryInstall | null>(null);
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
                    : setTrusting(install)
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
      {trusting && (
        <ConfirmDialog
          title={`Trust ${trusting.package_id}?`}
          question="Trusting this package lets its code run inside ComfyUI on this machine. Confirm only if you have reviewed this exact prepared package."
          detail={
            <dl className="confirm-facts">
              <div>
                <dt>Package</dt>
                <dd>{trusting.package_id}</dd>
              </div>
              <div>
                <dt>Version</dt>
                <dd>{trusting.package_version}</dd>
              </div>
              <div>
                <dt>Archive digest</dt>
                <dd>
                  <code>{trusting.archive_sha256}</code>
                </dd>
              </div>
            </dl>
          }
          confirmLabel="I reviewed this package - trust it"
          tone="trust"
          onCancel={() => setTrusting(null)}
          onConfirm={() => {
            const chosen = trusting;
            setTrusting(null);
            review.mutate({ id: chosen.id, trusted: true });
          }}
        />
      )}
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
