import { useState } from "react";
import { CatalogVariantChoice } from "./CatalogVariantChoice";
import { useMutation } from "@tanstack/react-query";
import { Download, Search } from "lucide-react";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import { catalogRoleFor, searchTermFor } from "./workflowAssetSearch";
import type {
  CatalogFileVariant,
  CatalogPreflight,
  CatalogModel,
  WorkflowAssetReference,
  WorkflowAssetReview,
  WorkflowSourceCandidate,
} from "./types";

type Selection = {
  reference_filename: string;
  install_plan_id: string;
  artifact_path: string;
};

/** The one file this asset resolves to, or nothing at all.
 *
 * A workflow reference answers to exactly one artifact. A plan holding more
 * than one means the repository was ranked rather than the file being read,
 * and binding the first of several would quietly install the rest.
 */
function exactArtifact(result: {
  selected_files: string[];
  can_install: boolean;
  install_plan?: { compatibility: string } | null;
}): { artifact: string } | { refusal: string } {
  // Each of these is a different thing to do about it, so each says which one
  // happened. Returning one null for all of them is what made a source the
  // author recorded look like a button that does nothing.
  if (!result.can_install) {
    return { refusal: "that source cannot be installed from here" };
  }
  if (result.install_plan?.compatibility !== "supported") {
    return { refusal: "that source is in a format this application cannot install" };
  }
  if (result.selected_files.length === 0) {
    return { refusal: "that source does not contain this file" };
  }
  if (result.selected_files.length > 1) {
    return {
      refusal:
        "that source holds several files and none of them is clearly this one; " +
        "search for it instead so the choice is explicit",
    };
  }
  return { artifact: result.selected_files[0] };
}

/** Install the model files a workflow needs, one explicit choice at a time.
 *
 * A workflow names filenames, not sources - so nothing is guessed here. For
 * each missing file the user searches the catalog, picks a candidate, and
 * the app preflights it into an immutable plan; the server then rebinds
 * everything from its own records and reports the exact cost before a byte
 * is queued. The browser only ever sends which plan artifact answers which
 * filename.
 */
export function WorkflowAssetInstaller({
  uiGraph,
  missing,
  onInstalled,
}: {
  uiGraph: Record<string, unknown>;
  missing: WorkflowAssetReference[];
  onInstalled?: (jobCount: number) => void;
}) {
  const [selections, setSelections] = useState<Record<string, Selection>>({});
  const [review, setReview] = useState<WorkflowAssetReview | null>(null);
  const [queuedCount, setQueuedCount] = useState<number | null>(null);

  const selectionList = missing
    .map((asset) => selections[asset.filename])
    .filter((selection): selection is Selection => Boolean(selection));

  const reviewMutation = useMutation({
    mutationFn: () => api.reviewWorkflowAssets(uiGraph, selectionList),
    onSuccess: setReview,
  });
  const installMutation = useMutation({
    mutationFn: (hash: string) => api.installWorkflowAssets(uiGraph, selectionList, hash),
    onSuccess: (jobs) => {
      setQueuedCount(jobs.length);
      onInstalled?.(jobs.length);
    },
  });

  const choose = (filename: string, selection: Selection | null) => {
    setReview(null);
    setQueuedCount(null);
    setSelections((current) => {
      const next = { ...current };
      if (selection) next[filename] = selection;
      else delete next[filename];
      return next;
    });
  };

  const error = reviewMutation.error ?? installMutation.error;
  if (missing.length === 0) return null;

  return (
    <section className="workflow-asset-installer">
      <h3>Model files this workflow needs</h3>
      <p className="package-review-note">
        Choose where each file comes from. Nothing downloads until you review
        the exact set and confirm it.
      </p>
      {error && <ErrorCallout message={(error as Error).message} />}
      <ul className="asset-install-list">
        {missing.map((asset) => (
          <AssetRow
            key={asset.filename}
            asset={asset}
            selection={selections[asset.filename] ?? null}
            onChoose={(selection) => choose(asset.filename, selection)}
          />
        ))}
      </ul>
      {review && (
        <div className="asset-install-review" role="status">
          <strong>
            {review.download_count} download{review.download_count === 1 ? "" : "s"}
            {" · "}
            {formatBytes(review.total_bytes)}
          </strong>
          <small>
            {review.assets.length} file{review.assets.length === 1 ? "" : "s"} bound to verified
            plans. Each download is independent: a finished one stays installed even if another
            fails.
          </small>
        </div>
      )}
      {queuedCount !== null && (
        <p className="package-review-note" role="status">
          Queued {queuedCount} download{queuedCount === 1 ? "" : "s"} - progress shows in the jobs
          panel. Re-check the workflow once they finish.
        </p>
      )}
      <div className="storage-actions">
        <button
          className="secondary"
          disabled={selectionList.length === 0 || reviewMutation.isPending}
          onClick={() => reviewMutation.mutate()}
        >
          {reviewMutation.isPending ? "Checking…" : `Review ${selectionList.length} selected`}
        </button>
        <button
          className="primary"
          disabled={!review || installMutation.isPending}
          onClick={() => review && installMutation.mutate(review.binding_plan_hash)}
        >
          <Download size={16} />
          {installMutation.isPending ? "Queueing…" : "Install these files"}
        </button>
      </div>
    </section>
  );
}

function AssetRow({
  asset,
  selection,
  onChoose,
}: {
  asset: WorkflowAssetReference;
  selection: Selection | null;
  onChoose: (selection: Selection | null) => void;
}) {
  const [query, setQuery] = useState(() => searchTermFor(asset.filename));
  const [results, setResults] = useState<CatalogModel[]>([]);
  const [source, setSource] = useState("civitai");
  // A blocked preflight that named more than one file behind the requested
  // name. Kept with the candidate that produced it, so choosing re-runs the
  // same check with the choice rather than starting over.
  const [ambiguous, setAmbiguous] = useState<
    { candidate: WorkflowSourceCandidate; variants: CatalogFileVariant[] } | null
  >(null);

  // Why a preflight that raised no error still produced nothing to select. It
  // is not an error - the request succeeded - so it has nowhere else to go, and
  // without it the button reads as broken rather than as unable.
  const [refusal, setRefusal] = useState<string | null>(null);

  const search = useMutation({
    mutationFn: () => api.catalog(query, catalogRoleFor(), "downloads", null, {}, source),
    onSuccess: (page) => setResults(page.items.slice(0, 5)),
  });

  // Preflighting a candidate produces the immutable plan the binding needs;
  // the plan is what carries hashes and sizes, never the browser.
  /** What every preflight result means, whichever path asked for it.
   *
   * An ambiguous version is not an error: the version is fine and the request
   * was under-specified, which is answerable right where it was refused. Both
   * the searched result and the author's recorded candidate arrive here so a
   * manually found version is as answerable as a recorded one.
   */
  const handlePreflight = ({
    result,
    candidate,
  }: {
    result: CatalogPreflight;
    candidate: WorkflowSourceCandidate;
  }) => {
    const wanted = candidate.filename || asset.filename;
    const variants = result.file_variants?.[wanted] ?? [];
    if (!result.can_install && variants.length > 1) {
      setAmbiguous({ candidate, variants });
      return;
    }
    setAmbiguous(null);
    const planId = result.install_plan?.id;
    const outcome = exactArtifact(result);
    if (!planId) {
      setRefusal("that source produced no install plan");
      return;
    }
    if ("refusal" in outcome) {
      setRefusal(outcome.refusal);
      return;
    }
    setRefusal(null);
    onChoose({
      reference_filename: asset.filename,
      install_plan_id: planId,
      artifact_path: outcome.artifact,
    });
  };

  const preflight = useMutation({
    mutationFn: (model: CatalogModel) =>
      api.catalogPreflight(
        model.remote_id,
        asset.kind === "lora" ? "image" : "image",
        model.required_runtime ?? "comfyui",
        model.provider === "civitai" ? model.remote_id : "main",
        // Name the file the workflow asked for. Leaving this empty let the
        // repository's own bundle be ranked instead, which is how one text
        // encoder turned into a four-file, 19GB install.
        [asset.filename],
        asset.kind === "lora" ? "lora" : null,
        null,
        model.provider,
        asset.kind,
      ).then((result) => ({
        result,
        // A searched result stands in for a candidate so both paths answer an
        // ambiguous version the same way. Nothing else distinguishes them.
        candidate: {
          provider: model.provider,
          remote_id: model.remote_id,
          revision: model.provider === "civitai" ? model.remote_id : "main",
          filename: asset.filename,
          url: model.remote_id,
        } as WorkflowSourceCandidate,
      })),
    onMutate: () => setRefusal(null),
    onSuccess: (payload) => handlePreflight(payload),
  });

  // The author recorded where this exact file came from. Searching a catalog
  // for a filename frequently finds nothing - a repository file is not
  // indexed by its path - so the recorded source is offered first, and it
  // still goes through the same preflight that produces an immutable plan.
  const recorded = asset.source_candidates;

  const preflightCandidate = useMutation({
    mutationFn: ({
      candidate,
      fileIds = [],
    }: {
      candidate: WorkflowSourceCandidate;
      fileIds?: string[];
    }) =>
      api.catalogPreflight(
        candidate.remote_id,
        catalogRoleFor(),
        "comfyui",
        candidate.revision ?? "main",
        // Naming the file the author pointed at keeps a multi-file repository
        // from resolving to whichever file the catalog would have picked.
        // Exactly one identity, in exactly one form: the server refuses a
        // request that names a file both ways, and an exact id is the answer
        // to a filename that could not settle the choice.
        fileIds.length > 0 ? [] : [candidate.filename || asset.filename],
        asset.kind === "lora" ? "lora" : null,
        null,
        candidate.provider,
        asset.kind,
        fileIds,
      ).then((result) => ({ result, candidate })),
    onMutate: () => setRefusal(null),
    onSuccess: (payload) => handlePreflight(payload),
  });

  return (
    <li className="asset-install-row">
      <span>
        <strong>{asset.filename}</strong>
        <small>{asset.kind}</small>
      </span>
      {selection ? (
        <span className="asset-install-chosen">
          <span className="badge likely">Selected</span>
          <small>{selection.artifact_path}</small>
          <button className="secondary compact-button" onClick={() => onChoose(null)}>
            Change
          </button>
        </span>
      ) : (
        <div className="asset-install-search">
          {recorded.length > 0 && (
            <div className="asset-install-recorded">
              <small>Recorded by this workflow&rsquo;s author</small>
              <ul>
                {recorded.map((candidate) => (
                  <li key={candidate.url}>
                    <button
                      className="secondary compact-button"
                      disabled={preflightCandidate.isPending}
                      onClick={() => preflightCandidate.mutate({ candidate })}
                    >
                      {candidate.filename ?? candidate.remote_id}
                    </button>
                    <small>{candidate.provider}</small>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {ambiguous && (
            <CatalogVariantChoice
              filename={ambiguous.candidate.filename || asset.filename}
              variants={ambiguous.variants}
              busy={preflightCandidate.isPending}
              onChoose={(sourceFileId) =>
                preflightCandidate.mutate({
                  candidate: ambiguous.candidate,
                  fileIds: [sourceFileId],
                })
              }
            />
          )}
          <select
            aria-label={`Source for ${asset.filename}`}
            value={source}
            onChange={(event) => setSource(event.target.value)}
          >
            <option value="civitai">CivitAI</option>
            <option value="huggingface">Hugging Face</option>
          </select>
          <input
            aria-label={`Search for ${asset.filename}`}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button
            className="secondary compact-button"
            disabled={search.isPending}
            onClick={() => search.mutate()}
          >
            <Search size={14} />
            {search.isPending ? "Searching…" : "Search"}
          </button>
          {(search.error || preflight.error || preflightCandidate.error) && (
            <small role="alert">
              {((search.error ?? preflight.error ?? preflightCandidate.error) as Error).message}
            </small>
          )}
          {refusal && !search.error && !preflight.error && !preflightCandidate.error && (
            <small role="alert">{refusal}</small>
          )}
          <ul className="asset-install-candidates">
            {results.map((model) => (
              <li key={`${model.provider}:${model.remote_id}`}>
                <button
                  className="secondary compact-button"
                  disabled={preflight.isPending}
                  onClick={() => preflight.mutate(model)}
                >
                  {model.name}
                </button>
                <small>
                  {model.author ?? model.provider}
                  {model.total_size_bytes ? ` · ${formatBytes(model.total_size_bytes)}` : ""}
                </small>
              </li>
            ))}
          </ul>
        </div>
      )}
    </li>
  );
}
