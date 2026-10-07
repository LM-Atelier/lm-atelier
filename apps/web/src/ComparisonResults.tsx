import { useQuery } from "@tanstack/react-query";
import { ShieldedMedia } from "./ShieldedMedia";
import type { ReactNode, Ref } from "react";
import { api } from "./api";
import { ComparisonBlindReview } from "./ComparisonBlindReview";
import { ComparisonChoiceSummary, ComparisonEstimate, ComparisonRefusals, ComparisonSettings } from "./ComparisonCheckResult";
import { ComparisonKeepRecipe } from "./ComparisonKeepRecipe";
import { ComparisonPreference } from "./ComparisonPreference";
import { ErrorCallout } from "./ErrorCallout";
import { GenerationDetails } from "./GenerationDetails";
import { comparisonFailure, comparisonIsWorking, keptPicture, TRIAL_STATUS_TEXT, trialIsWorking } from "./generationComparison";
import type { ExperimentArm, ExperimentTrial, GenerationExperiment } from "./generationExperimentTypes";
import { jobProgressFraction, jobProgressText } from "./jobProgress";
import { artifactSource } from "./messageMedia";
import type { Job } from "./types";
import "./GenerationComparisonView.css";

function TrialPicture({ experimentId, arm, trial, job, video }: {
  experimentId: string; arm: ExperimentArm; trial: ExperimentTrial; job: Job | undefined;
  /** The comparison made videos rather than pictures. */
  video: boolean;
}) {
  const noun = video ? "video" : "picture";
  const finished = trial.status === "complete" && Boolean(trial.run_id);
  const run = useQuery({
    queryKey: ["generation-experiments", experimentId, "run", trial.run_id],
    queryFn: () => api.run(trial.run_id as string),
    enabled: finished,
    staleTime: Infinity,
  });
  const picture = run.data ? keptPicture(run.data, trial, video ? "video" : "image") : null;
  const fraction = job ? jobProgressFraction(job) : null;
  return <>
    <p>{trial.status ? TRIAL_STATUS_TEXT[trial.status] : "Not started"}</p>
    {trialIsWorking(trial.status) && job && <div className="progress-track" role="progressbar"
      aria-label={`${arm.label} progress`} aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}>
      <div className={fraction === null ? "indeterminate" : undefined}
        style={fraction === null ? undefined : { width: `${fraction * 100}%` }} />
    </div>}
    {trialIsWorking(trial.status) && job && <small>{jobProgressText(job)}</small>}
    {finished && run.isPending && <p role="status">{`Loading the ${noun}…`}</p>}
    {finished && run.isError && <p role="alert">{`This ${noun} could not be loaded.`}</p>}
    {picture && <figure className="comparison-picture">
      {video ? <ShieldedMedia kind="video">
        {/* Generated media has no caption track to point at, and an empty one would claim an affordance that is not there. */}
        {/* eslint-disable-next-line jsx-a11y-x/media-has-caption */}
        <video src={artifactSource(picture) ?? undefined} controls preload="metadata" aria-label={`Made by ${arm.label}`} />
      </ShieldedMedia>
        : <ShieldedMedia kind="image"><img src={artifactSource(picture) ?? undefined} alt={`Made by ${arm.label}`} /></ShieldedMedia>}
    </figure>}
    {finished && run.data && !picture && <p>{`No ${noun} was recorded for this choice.`}</p>}
    {run.data && picture && <GenerationDetails provenance={run.data.provenance_json} />}
  </>;
}

/** An accepted comparison: start it, then both results side by side as they are made. */
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
  const video = experiment.operation === "text_to_video";
  const nouns = video ? "videos" : "pictures";
  return <section className="comparison-results" aria-labelledby="comparison-results-heading">
    <h2 id="comparison-results-heading" ref={headingRef} tabIndex={-1}>{experiment.name}</h2>
    {experiment.state === "started" && !experiment.blind_pending && <p role="status">{ready === trials.length
      ? `Both ${nouns} are ready` : `${ready} of ${trials.length} ${nouns} ready`}</p>}
    {experiment.state === "ready" && <>
      <ComparisonEstimate estimate={experiment.estimate} nouns={nouns} />
      <button type="button" className="primary" aria-disabled={starting}
        onClick={() => { if (!starting) onStart(experiment); }}>{starting ? "Starting…" : `Make both ${nouns}`}</button>
    </>}
    {failure && <><ErrorCallout message={failure.message} />
      <ComparisonRefusals refusals={failure.refusals} labels={labels} /></>}
    {children}
    {/* While blind, the pictures are shown only by position, and nothing on the page says which choice made which. */}
    {experiment.state === "started" && experiment.blind_pending && <ComparisonBlindReview experiment={experiment} />}
    {!(experiment.state === "started" && experiment.blind_pending) && <div className="comparison-columns">{experiment.arms.map((arm) => <section key={arm.id}
      aria-labelledby={`comparison-arm-${arm.ordinal}`}>
      <h3 id={`comparison-arm-${arm.ordinal}`}>{arm.label}</h3>
      {arm.trials.map((trial) => <TrialPicture key={trial.id} experimentId={experiment.id} arm={arm} trial={trial}
        job={(jobs.data ?? []).find((job) => job.id === trial.job_id)} video={video} />)}
      {/* Kept once its picture is seen: the reason to keep a setup is what it made. */}
      {arm.trials.some((trial) => trial.status === "complete")
        && <ComparisonKeepRecipe experimentId={experiment.id} operation={experiment.operation} arm={arm}
          onOpenChat={onOpenChat} />}
      <ComparisonChoiceSummary arm={arm} seed={arm.trials[0]?.seed ?? null} />
    </section>)}</div>}
    {/* Said once there is a picture to say it of. */}
    {experiment.state === "started" && ready > 0 && <ComparisonPreference experiment={experiment} />}
    <ComparisonSettings arms={experiment.arms} />
    <button type="button" className="secondary" onClick={onNew}>New comparison</button>
  </section>;
}
