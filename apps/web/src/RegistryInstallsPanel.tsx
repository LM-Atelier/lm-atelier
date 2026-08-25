import { useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import { activationErrorDescription } from "./registryPreparationErrors";
import type { RegistryInstall, RegistryInstallReview } from "./types";

/** Prepared Registry packages and the two decisions each one waits for.
 *
 * Preparation only stages files; nothing here runs until the exact package is
 * explicitly trusted, and nothing loads into ComfyUI until it is separately
 * activated. Revoking trust also deactivates. Every action needs the media
 * worker stopped, and the server refuses with a stable code when it is not.
 */
export function RegistryInstallsPanel() {
  const client = useQueryClient();
  const installs = useQuery({
    queryKey: ["registry-installs"],
    queryFn: api.registryInstalls,
    refetchInterval: 3_000,
  });
  const refresh = () => void client.invalidateQueries({ queryKey: ["registry-installs"] });
  const [trusting, setTrusting] = useState<RegistryInstall | null>(null);
  const [removing, setRemoving] = useState<RegistryInstall | null>(null);
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
  const renew = useMutation({
    mutationFn: (id: string) => api.renewRegistryInstall(id),
    onSuccess: () => {
      refresh();
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.removeRegistryInstall(id),
    onSuccess: refresh,
  });
  const error = (review.error ?? activate.error ?? deactivate.error ?? renew.error ?? remove.error) as
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
            {install.disk_status !== "ready" && (
              <span className="badge advanced_import">Files missing</span>
            )}
            <span>
              <strong>{install.package_id}</strong>
              <small>
                {install.package_version}
                {` · ${install.node_types.length} node type${install.node_types.length === 1 ? "" : "s"}`}
              </small>
              {install.disk_status !== "ready" && (
                <small>This prepared package is incomplete. Remove it, then prepare it again.</small>
              )}
            </span>
            <details>
              <summary>Identity</summary>
              <pre>{identityLines(install)}</pre>
            </details>
            <span className="row-actions">
              {(install.trusted || install.disk_status === "ready") && (
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
              )}
              {install.trusted && !install.active && install.disk_status === "ready" && (
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
              {!install.active && install.node_files_present && (
                <button
                  className="secondary compact-button"
                  disabled={renew.isPending && renew.variables === install.id}
                  onClick={() => renew.mutate(install.id)}
                >
                  Refresh dependencies
                </button>
              )}
              {!install.active && (
                <button
                  className="secondary compact-button danger"
                  disabled={remove.isPending && remove.variables === install.id}
                  onClick={() => setRemoving(install)}
                >
                  Remove
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
              {reviewFacts(trusting.review)}
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
      {removing && (
        <ConfirmDialog
          title={`Remove ${removing.package_id}?`}
          question="This removes the prepared package record and its managed package files. You can prepare the package again later."
          detail={
            <dl className="confirm-facts">
              <div>
                <dt>Package</dt>
                <dd>{removing.package_id}</dd>
              </div>
              <div>
                <dt>Version</dt>
                <dd>{removing.package_version}</dd>
              </div>
            </dl>
          }
          confirmLabel="Remove prepared package"
          confirmDisabled={remove.isPending}
          onCancel={() => setRemoving(null)}
          onConfirm={() => {
            const chosen = removing;
            setRemoving(null);
            remove.mutate(chosen.id);
          }}
        />
      )}
    </section>
  );
}

/** The findings that decide the answer, and silence when there are none.
 *
 * Trust is what lets this code run, so the dialog names the things that
 * actually change the decision: code that runs on install, code that runs at
 * startup, and compiled binaries nobody can read. A package with none of
 * those says so once rather than listing three empty rows, and a package
 * prepared before this record existed says nothing at all - "not looked at"
 * is a different claim from "nothing found".
 */
function reviewFacts(review: RegistryInstallReview | null): ReactNode {
  if (!review) return null;
  const findings: [string, string[]][] = [
    ["Runs on install", review.install_scripts],
    ["Runs at startup", review.startup_hooks],
    ["Compiled binaries", review.native_files],
    ["Declares dependencies", review.dependency_manifests],
  ];
  const present = findings.filter(([, paths]) => paths.length > 0);
  return (
    <>
      <div>
        <dt>Contents</dt>
        <dd>
          {review.file_count} files, {formatBytes(review.expanded_bytes)},{" "}
          {review.python_file_count} of them Python
        </dd>
      </div>
      {present.length === 0 && (
        <div>
          <dt>Findings</dt>
          <dd>No install scripts, startup hooks, or compiled binaries.</dd>
        </div>
      )}
      {present.map(([label, paths]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>
            {paths.map((path) => (
              <code key={path}>{path}</code>
            ))}
          </dd>
        </div>
      ))}
      {review.registry_warnings.length > 0 && (
        <div>
          <dt>Needs review because</dt>
          <dd>{review.registry_warnings.map((warning) => warning.replaceAll("_", " ")).join(", ")}</dd>
        </div>
      )}
    </>
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
