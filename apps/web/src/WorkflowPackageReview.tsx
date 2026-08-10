import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { CircleAlert, CircleCheck, PackageSearch } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { preparationErrorDescription } from "./registryPreparationErrors";
import { WorkflowAssetInstaller } from "./WorkflowAssetInstaller";
import type { WorkflowPackageAnalysis } from "./types";

const ISSUE_DESCRIPTIONS: Record<string, string> = {
  blocked_asset_format: "References a model format this app never loads",
  conflicting_custom_node_versions: "Two nodes pin different versions of one package",
  dangling_link: "Has links that connect to nothing",
  remote_url_reference: "Reaches out to a remote URL",
  unsafe_asset_reference: "References a file outside the model folders",
  unsupported_asset_format: "References a model format this app cannot verify",
  unidentified_custom_node_package: "Uses custom nodes with no declared package",
  unresolved_custom_node_package: "Needs a package version this machine does not have installed",
  unversioned_custom_node_package: "Uses a package without a pinned version",
  // Installed and trusted already. What is missing is the record of which nodes
  // were reviewed, so the remedy is to read that revision again rather than to
  // go and fetch anything - which is what the wording has to convey, because
  // the two states are indistinguishable from the outside otherwise.
  custom_node_package_awaiting_review:
    "Uses an installed package whose review did not record which nodes it provides - review it again to confirm",
};

function installableAssets(analysis: WorkflowPackageAnalysis) {
  return analysis.asset_references.filter(
    (asset) => !asset.present_locally && asset.policy === "supported",
  );
}

function issueDescription(code: string): string {
  return ISSUE_DESCRIPTIONS[code] ?? code.replaceAll("_", " ");
}

const INVENTORY_DEPENDENT_ISSUES = new Set([
  "unidentified_custom_node_package",
  "unresolved_custom_node_package",
  // Whether a package resolves is read against the runtime's node inventory, so
  // without one every package looks unresolved and this would tell people to
  // re-review packages that are fine.
  "custom_node_package_awaiting_review",
]);

/** Review a raw ComfyUI package before anything is imported or trusted.
 *
 * Everything shown comes from the analyzer report; the only gate this surface
 * obeys is `dependencies_resolved`, computed server-side. Nothing here can
 * install, trust, or activate - the report has to be resolved first, and
 * resolution lives in a later backend step, not in this dialog.
 */
export function WorkflowPackageReview({
  analysis,
  fileName,
  uiGraph,
  onImported,
  onClose,
}: {
  analysis: WorkflowPackageAnalysis;
  fileName: string;
  uiGraph?: Record<string, unknown>;
  onImported?: () => void;
  onClose: () => void;
}) {
  const missingKnown = analysis.node_inventory_available;
  const visibleIssues = missingKnown
    ? analysis.issues
    : analysis.issues.filter((issue) => !INVENTORY_DEPENDENT_ISSUES.has(issue.code));
  const [queuedPackages, setQueuedPackages] = useState<string[]>([]);
  const [importName, setImportName] = useState(fileName.replace(/\.json$/i, ""));
  // The analyzer's guess prefills the form; the user confirms, nothing is
  // silently decided. Loading an image suggests the image-conditioned twin.
  const loadsImage = analysis.required_node_types.some((type) =>
    type.replaceAll(/[\s_]/g, "").toLowerCase().includes("loadimage"));
  const [operation, setOperation] = useState(
    analysis.operation_guess === "video"
      ? (loadsImage ? "image_to_video" : "text_to_video")
      : (loadsImage ? "image_to_image" : "text_to_image"),
  );
  const ensureDraft = () => api.ensureWorkflowPackageDraft({
    ui_graph: uiGraph ?? {},
    name: importName.trim(),
    operation,
  });
  const importWorkflow = useMutation({
    mutationFn: async () => {
      const draft = await ensureDraft();
      if (!draft.current_revision_id) {
        throw new Error("The workflow draft has no current revision.");
      }
      return api.importWorkflowPackage({
        ui_graph: uiGraph ?? {},
        name: importName.trim(),
        operation,
        draft_workflow_id: draft.id,
        draft_revision_id: draft.current_revision_id,
      });
    },
    onSuccess: () => onImported?.(),
  });
  const prepare = useMutation({
    mutationFn: ({ packageId, version, sourceGraph }: {
      packageId: string;
      version: string;
      sourceGraph: Record<string, unknown>;
    }) => ensureDraft().then((draft) => {
      if (!draft.current_revision_id) {
        throw new Error("The workflow draft has no current revision.");
      }
      return api.prepareWorkflowPackage(
        packageId,
        version,
        sourceGraph,
        draft.current_revision_id,
      );
    }),
    onSuccess: (_job, variables) =>
      setQueuedPackages((current) => [...current, variables.packageId]),
  });
  const prepareError = prepare.error as (Error & { code?: string }) | null;
  return (
    <AccessibleDialog
      title="Review workflow package"
      eyebrow="Nothing is imported yet"
      closeLabel="Close package review"
      onClose={onClose}
      className="workflow-package-review"
    >
      <header className="package-review-summary">
        {analysis.ready
          ? <CircleCheck aria-hidden="true" className="ready" />
          : <CircleAlert aria-hidden="true" className="unresolved" />}
        <div>
          <strong>{fileName}</strong>
          <small>
            ComfyUI {analysis.format_version}
            {analysis.frontend_version ? ` · frontend ${analysis.frontend_version}` : ""}
            {` · ${analysis.node_count} node${analysis.node_count === 1 ? "" : "s"}`}
            {analysis.subgraph_count > 0 ? ` · ${analysis.subgraph_count} subgraph${analysis.subgraph_count === 1 ? "" : "s"}` : ""}
          </small>
          <p>
            {analysis.ready
              ? "Everything this workflow needs is present. Name it and import below; it arrives untrusted."
              : "This workflow cannot be imported or trusted until everything below is resolved."}
          </p>
        </div>
      </header>
      {!missingKnown && (
        <p className="package-review-note">
          The media runtime is not running, so node availability is unknown.
          Start it to see which nodes are actually missing.
        </p>
      )}
      {analysis.missing_nodes.length > 0 && missingKnown && (
        <section>
          <h3>Missing node types</h3>
          <ul>
            {analysis.missing_nodes.map((missing) => (
              <li key={missing.node_type}>
                <code>{missing.node_type}</code>
                <small>
                  {` · ${missing.count} node${missing.count === 1 ? "" : "s"}`}
                  {missing.package_id ? ` · from ${missing.package_id}` : ""}
                </small>
              </li>
            ))}
          </ul>
        </section>
      )}
      {analysis.custom_packages.length > 0 && (
        <section>
          <h3>Custom node packages</h3>
          <ul>
            {analysis.custom_packages.map((pkg) => (
              <li key={pkg.package_id}>
                <PackageSearch size={14} aria-hidden="true" />
                <code>{pkg.package_id}</code>
                <small>
                  {pkg.versions.length > 0 ? ` ${pkg.versions.join(", ")}` : " no pinned version"}
                  {` · ${pkg.node_types.length} node type${pkg.node_types.length === 1 ? "" : "s"}`}
                </small>
                <span className={`badge ${pkg.locally_resolved ? "likely" : "advanced_import"}`}>
                  {pkg.locally_resolved ? "installed" : "not installed"}
                </span>
                {uiGraph && !pkg.locally_resolved && pkg.versions.length === 1 && (
                  queuedPackages.includes(pkg.package_id)
                    ? <small>Preparation queued - progress shows in the jobs panel. The result stays inactive and untrusted until reviewed.</small>
                    : (
                      <button
                        type="button"
                        className="secondary compact-button"
                        disabled={prepare.isPending}
                        onClick={() => prepare.mutate({
                          packageId: pkg.package_id,
                          version: pkg.versions[0],
                          sourceGraph: uiGraph,
                        })}
                      >
                        Prepare {pkg.versions[0]}
                      </button>
                    )
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
      {/* Only supported formats are offered: a blocked or unverifiable
          format is a refusal, not something to go fetch. */}
      {uiGraph && installableAssets(analysis).length > 0 && (
        <WorkflowAssetInstaller uiGraph={uiGraph} missing={installableAssets(analysis)} />
      )}
      {analysis.asset_references.length > 0 && (
        <section>
          <h3>Model files it references</h3>
          <ul>
            {analysis.asset_references.map((asset) => (
              <li key={asset.filename}>
                <code>{asset.filename}</code>
                <small>{` ${asset.kind}`}</small>
                <span className={`badge ${asset.policy !== "supported" ? "advanced_import" : asset.present_locally ? "likely" : "advanced_import"}`}>
                  {asset.policy !== "supported" ? asset.policy : asset.present_locally ? "installed" : "not installed"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
      {/* Authors usually name a model the way it is titled on its page, not
          the way it is named on disk, so these links cannot be matched to a
          file automatically. Showing them is still the difference between a
          findable model and a dead end. */}
      {analysis.source_candidates.length > 0 && (
        <section>
          <h3>Sources this workflow mentions</h3>
          <p className="package-review-note">
            The author recorded these, but did not say which file each one is.
            Open one to see what it is, then pick it above for the file it
            belongs to.
          </p>
          <ul>
            {analysis.source_candidates.map((candidate) => (
              <li key={candidate.url}>
                <a href={candidate.url} target="_blank" rel="noreferrer noopener">
                  {candidate.filename ?? candidate.remote_id}
                </a>
                <small>{` ${candidate.provider}`}</small>
              </li>
            ))}
          </ul>
        </section>
      )}
      {visibleIssues.length > 0 && (
        <section>
          <h3>Findings</h3>
          <ul>
            {visibleIssues.map((issue) => (
              <li key={issue.code}>
                {issue.severity === "blocking"
                  ? <span className="badge advanced_import">blocking</span>
                  : <span className="badge likely">advisory</span>}
                {issueDescription(issue.code)}
                <small>
                  {` · ${issue.count} place${issue.count === 1 ? "" : "s"}`}
                  {issue.node_types.length > 0 ? ` · ${issue.node_types.join(", ")}` : ""}
                </small>
              </li>
            ))}
          </ul>
        </section>
      )}
      {prepareError && (
        <p role="alert" className="package-review-note">
          {prepareError.code ? preparationErrorDescription(prepareError.code) : prepareError.message}
        </p>
      )}
      {analysis.ready && uiGraph && onImported && (
        <div className="package-import-form">
          <label>
            Workflow name
            <input
              value={importName}
              maxLength={240}
              onChange={(event) => setImportName(event.target.value)}
            />
          </label>
          <label>
            Operation
            <select value={operation} onChange={(event) => setOperation(event.target.value)}>
              <option value="text_to_image">Text to image</option>
              <option value="image_to_image">Image to image</option>
              <option value="text_to_video">Text to video</option>
              <option value="image_to_video">Image to video</option>
            </select>
          </label>
          {importWorkflow.error && (
            <p role="alert" className="package-review-note">{importWorkflow.error.message}</p>
          )}
        </div>
      )}
      <footer>
        <button className="secondary" onClick={onClose}>Close</button>
        {analysis.ready && uiGraph && onImported && (
          <button
            className="primary"
            disabled={!importName.trim() || importWorkflow.isPending}
            onClick={() => importWorkflow.mutate()}
          >
            {importWorkflow.isPending ? "Importing…" : "Import workflow"}
          </button>
        )}
      </footer>
    </AccessibleDialog>
  );
}
