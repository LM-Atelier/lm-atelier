import { useQuery } from "@tanstack/react-query";
import type { ReactNode, Ref } from "react";
import { api } from "./api";
import { ComparisonChoiceSummary, ComparisonEstimate, ComparisonRefusals, ComparisonSettings } from "./ComparisonCheckResult";
import { ComparisonKeepRecipe } from "./ComparisonKeepRecipe";
import { ErrorCallout } from "./ErrorCallout";
import { GenerationDetails } from "./GenerationDetails";
import { comparisonFailure, comparisonIsWorking, keptPicture, trialIsWorking } from "./generationComparison";
import type { ExperimentArm, ExperimentTrial, GenerationExperiment, TrialWorkStatus } from "./generationExperimentTypes";
import { jobProgressFraction, jobProgressText } from "./jobProgress";
import { artifactSource } from "./messageMedia";
import type { Job } from "./types";
import "./GenerationComparisonView.css";

const STATUS_TEXT: Record<TrialWorkStatus, string> = {
  queued: "Waiting to start", running: "Being made", paused: "Paused", blocked: "Waiting",
  complete: "Finished", failed: "Failed", cancelled: "Stopped", interrupted: "Interrupted",
  removed: "Its work was removed",
};

function TrialPicture({ experimentId, arm, trial, job }: {
  experimentId: string; arm: ExperimentArm; trial: ExperimentTrial; job: Job | undefined;
}) {
  const finished = trial.status === "complete" && Boolean(trial.run_id);
  const run = useQuery({
    queryKey: ["generation-experiments", experimentId, "run", trial.run_id],
    queryFn: () => api.run(trial.run_id as string),
    enabled: finished,
    staleTime: Infinity,
  });
  const picture = run.data ? keptPicture(run.data, trial) : null;
  const fraction = job ? jobProgressFraction(job) : null;
  return <>
    <p>{trial.status ? STATUS_TEXT[trial.status] : "Not started"}</p>
    {trialIsWorking(trial.status) && job && <div className="progress-track" role="progressbar"
      aria-label={`${arm.label} progress`} aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}>
      <div className={fraction === null ? "indeterminate" : undefined}
        style={fraction === null ? undefined : { width: `${fraction * 100}%` }} />
    </div>}
    {trialIsWorking(trial.status) && job && <small>{jobProgressText(job)}</small>}
    {finished && run.isPending && <p role="status">Loading the picture…</p>}
    {finished && run.isError && <p role="alert">This picture could not be loaded.</p>}
    {picture && <figure className="comparison-picture">
      <img src={artifactSource(picture) ?? undefined} alt={`Made by ${arm.label}`} />
    </figure>}
    {finished && run.data && !picture && <p>No picture was recorded for this choice.</p>}
    {run.data && picture && <GenerationDetails provenance={run.data.provenance_json} />}
  </>;
}

/** An accepted comparison: start it, then both pictures side by side as they are made. */
export function ComparisonResults({ experimentId, onStart, starting, startError, onNew, onOpenChat, headingRef, children }: {
  experimentId: string;
  onStart: (experiment: GenerationExperiment) => void;
  starting: boolean;
  startError: unknown;
  onNew: () => void;
  /** Show a chat set up with a choice that was kept. */
  onOpenChat?: (chatId: string) => void;
  headingRef?: Ref<HTMLHeadingElement>;
  children?: ReactNode;
}) {
  const read = useQuery({
    queryKey: ["generation-experiments", experimentId],
    queryFn: async ({ signal }) => {
      const experiment = await api.generationExperiment(experimentId, signal);
      if (experiment.id !== experimentId) throw new Error("A different comparison was returned.");
      return experiment;
    },
    refetchInterval: (query) => comparisonIsWorking(query.state.data) ? 2_000 : false,
  });
  const working = comparisonIsWorking(read.data);
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, enabled: working });
  if (read.isPending) return <p role="status">Loading the comparison…</p>;
  if (read.isError) {
    const failure = comparisonFailure(read.error);
    return <section aria-labelledby="comparison-results-heading">
      <h2 id="comparison-results-heading" ref={headingRef} tabIndex={-1}>Comparison</h2>
      <ErrorCallout message={failure.message} />
      <button type="button" className="secondary" onClick={onNew}>New comparison</button>
    </section>;
  }
  const experiment = read.data;
  const trials = experiment.arms.flatMap((arm) => arm.trials);
  const ready = trials.filter((trial) => trial.status === "complete").length;
  const failure = startError ? comparisonFailure(startError) : null;
  const labels = experiment.arms.map((arm) => arm.label);
  return <section className="comparison-results" aria-labelledby="comparison-results-heading">
    <h2 id="comparison-results-heading" ref={headingRef} tabIndex={-1}>{experiment.name}</h2>
    {experiment.state === "started" && <p role="status">{ready === trials.length
      ? "Both pictures are ready" : `${ready} of ${trials.length} pictures ready`}</p>}
    {experiment.state === "ready" && <>
      <ComparisonEstimate estimate={experiment.estimate} />
      <button type="button" className="primary" aria-disabled={starting}
        onClick={() => { if (!starting) onStart(experiment); }}>{starting ? "Starting…" : "Make both pictures"}</button>
    </>}
    {failure && <><ErrorCallout message={failure.message} />
      <ComparisonRefusals refusals={failure.refusals} labels={labels} /></>}
    {children}
    <div className="comparison-columns">{experiment.arms.map((arm) => <section key={arm.id}
      aria-labelledby={`comparison-arm-${arm.ordinal}`}>
      <h3 id={`comparison-arm-${arm.ordinal}`}>{arm.label}</h3>
      {arm.trials.map((trial) => <TrialPicture key={trial.id} experimentId={experiment.id} arm={arm} trial={trial}
        job={(jobs.data ?? []).find((job) => job.id === trial.job_id)} />)}
      {/* Kept once its picture is seen: the reason to keep a setup is what it made. */}
      {arm.trials.some((trial) => trial.status === "complete")
        && <ComparisonKeepRecipe experimentId={experiment.id} arm={arm} onOpenChat={onOpenChat} />}
      <ComparisonChoiceSummary arm={arm} seed={arm.trials[0]?.seed ?? null} />
    </section>)}</div>
    <ComparisonSettings arms={experiment.arms} />
    <button type="button" className="secondary" onClick={onNew}>New comparison</button>
  </section>;
}
