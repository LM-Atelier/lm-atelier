import { CircleAlert, CircleCheck, PackageSearch } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import type { WorkflowPackageAnalysis } from "./types";

const ISSUE_DESCRIPTIONS: Record<string, string> = {
  blocked_asset_format: "References a model format this app never loads",
  conflicting_custom_node_versions: "Two nodes pin different versions of one package",
  dangling_link: "Has links that connect to nothing",
  remote_url_reference: "Reaches out to a remote URL",
  unsafe_asset_reference: "References a file outside the model folders",
  unsupported_asset_format: "References a model format this app cannot verify",
  unidentified_custom_node_package: "Uses custom nodes with no declared package",
  unversioned_custom_node_package: "Uses a package without a pinned version",
};

function issueDescription(code: string): string {
  return ISSUE_DESCRIPTIONS[code] ?? code.replaceAll("_", " ");
}

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
  onClose,
}: {
  analysis: WorkflowPackageAnalysis;
  fileName: string;
  onClose: () => void;
}) {
  const missingKnown = analysis.node_inventory_available;
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
              ? "Everything this workflow needs is present. Import and compilation follow as a separate step."
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
              </li>
            ))}
          </ul>
        </section>
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
      {analysis.issues.length > 0 && (
        <section>
          <h3>Findings</h3>
          <ul>
            {analysis.issues.map((issue) => (
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
      <footer>
        <button className="secondary" onClick={onClose}>Close</button>
      </footer>
    </AccessibleDialog>
  );
}
