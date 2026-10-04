import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ShieldedMedia } from "./ShieldedMedia";
import { ApiError, api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { blindPictureSource, comparisonFailure, TRIAL_STATUS_TEXT, trialIsWorking } from "./generationComparison";
import type { GenerationExperiment, GenerationExperimentBlindEvaluationCreate } from "./generationExperimentTypes";
import "./GenerationComparisonView.css";

const VIEW_KEY = "lm-atelier.generation-comparison.blind-view";

/** The viewing this tab has open for a comparison, so a reload shows the same order rather than drawing a new one. */
function keptView(experimentId: string): string | null {
  try {
    const kept: unknown = JSON.parse(window.sessionStorage.getItem(VIEW_KEY) ?? "null");
    if (kept && typeof kept === "object" && (kept as { experimentId?: unknown }).experimentId === experimentId) {
      const viewId = (kept as { viewId?: unknown }).viewId;
      return typeof viewId === "string" ? viewId : null;
    }
  } catch {
    // Unreadable storage only means a new viewing, with its own order.
  }
  return null;
}

function keepView(experimentId: string, viewId: string | null) {
  try {
    if (viewId) window.sessionStorage.setItem(VIEW_KEY, JSON.stringify({ experimentId, viewId }));
    else window.sessionStorage.removeItem(VIEW_KEY);
  } catch {
    // Nothing to keep it in: a reload opens a new viewing instead.
  }
}

/** A blind comparison's pictures by where they are shown, until the preference is said.
 *
 * Nothing here names a choice: the pictures come from the viewing by position,
 * under no name of their own, with only their position as their text. Saying
 * the preference reveals which choice made each, and the page then shows the
 * comparison with its choices named.
 */
export function ComparisonBlindReview({ experiment }: { experiment: GenerationExperiment }) {
  const client = useQueryClient();
  const view = useQuery({
    queryKey: ["generation-experiments", experiment.id, "blind-view"],
    queryFn: async ({ signal }) => {
      const kept = keptView(experiment.id);
      if (kept) {
        try {
          return await api.blindView(experiment.id, kept, signal);
        } catch (error) {
          if (!(error instanceof ApiError && error.status === 404)) throw error;
        }
      }
      const opened = await api.openBlindView(experiment.id);
      keepView(experiment.id, opened.id);
      return opened;
    },
    refetchInterval: (query) => query.state.data?.positions.some((position) =>
      !position.ready && (position.status === null || trialIsWorking(position.status))) ? 2_000 : false,
  });
  const say = useMutation({
    mutationFn: ({ viewId, said }: { viewId: string; said: GenerationExperimentBlindEvaluationCreate }) =>
      api.sayBlindPreference(experiment.id, viewId, said),
    onSuccess: () => {
      keepView(experiment.id, null);
      void client.invalidateQueries({ queryKey: ["generation-experiments", experiment.id], exact: true });
    },
  });
  if (view.isPending) return <p role="status">Opening the pictures…</p>;
  if (view.isError) return <ErrorCallout message={comparisonFailure(view.error).message} />;
  const opened = view.data;
  const ready = opened.positions.filter((position) => position.ready).length;
  const sayIt = (said: GenerationExperimentBlindEvaluationCreate) => {
    if (!say.isPending && !say.isSuccess) say.mutate({ viewId: opened.id, said });
  };
  const waiting = say.isPending || say.isSuccess || ready === 0;
  return <section className="comparison-blind" aria-labelledby="comparison-blind-heading">
    <h3 id="comparison-blind-heading">Compared blind</h3>
    <p>Which choice made each picture stays hidden until you say which you prefer.</p>
    <p role="status">{ready === opened.positions.length ? "Both pictures are ready" : `${ready} of ${opened.positions.length} pictures ready`}</p>
    <div className="comparison-columns">{opened.positions.map((position) => <figure key={position.position} className="comparison-picture">
      {position.ready
        ? <ShieldedMedia kind="image"><img src={blindPictureSource(experiment.id, opened.id, position.position)} alt={`Shown at position ${position.position}`} /></ShieldedMedia>
        : <p>{position.status ? TRIAL_STATUS_TEXT[position.status] : "Not started"}</p>}
      <figcaption>Picture {position.position}</figcaption>
    </figure>)}</div>
    <fieldset className="comparison-preference">
      <legend>Which do you prefer?</legend>
      <div className="row-actions">
        {opened.positions.map((position) => <button key={position.position} type="button" className="secondary"
          aria-disabled={waiting || !position.ready}
          onClick={() => { if (position.ready) sayIt({ preference: "preferred", position: position.position }); }}>
          Prefer picture {position.position}</button>)}
        <button type="button" className="secondary" aria-disabled={waiting} onClick={() => { if (ready) sayIt({ preference: "tied" }); }}>Tie</button>
        <button type="button" className="secondary" aria-disabled={waiting}
          onClick={() => { if (ready) sayIt({ preference: "unsuitable" }); }}>Neither suits</button>
      </div>
    </fieldset>
    {say.isPending && <p role="status">Keeping your preference…</p>}
    <ErrorCallout message={say.error ? comparisonFailure(say.error).message : null} />
  </section>;
}
