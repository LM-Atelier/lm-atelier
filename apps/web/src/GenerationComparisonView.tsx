import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { ComparisonCheckResult, ComparisonRefusals } from "./ComparisonCheckResult";
import { ComparisonChoiceFields } from "./ComparisonChoiceFields";
import { ComparisonResults } from "./ComparisonResults";
import { ComparisonSharedFields } from "./ComparisonSharedFields";
import { ErrorCallout } from "./ErrorCallout";
import {
  comparisonFailure,
  comparisonRequest,
  EMPTY_COMPARISON,
  sameComparisonRequest,
  sharedPresetIds,
  type ComparisonChoiceDraft,
  type ComparisonDraft,
} from "./generationComparison";
import type { GenerationExperiment } from "./generationExperimentTypes";
import { useConfirm } from "./useConfirm";
import { useGenerationComparison } from "./useGenerationComparison";
import "./GenerationComparisonView.css";

function useShape(revisionId: string) {
  return useQuery({
    queryKey: ["workflow-revision", revisionId, "output-geometry"],
    queryFn: () => api.workflowRevisionOutputGeometry(revisionId),
    enabled: Boolean(revisionId),
  });
}

/** Compare two generation choices against one prompt, then see both pictures side by side. */
export function GenerationComparisonView({ onOpenChat }: {
  /** Show a chat set up with a choice that was kept. */
  onOpenChat?: (chatId: string) => void;
} = {}) {
  const comparison = useGenerationComparison();
  const [draft, setDraft] = useState<ComparisonDraft>(EMPTY_COMPARISON);
  const [problems, setProblems] = useState<string[]>([]);
  const [confirmation, confirm] = useConfirm();
  const checkHeading = useRef<HTMLHeadingElement>(null);
  const resultsHeading = useRef<HTMLHeadingElement>(null);
  const firstLabel = useRef<HTMLInputElement>(null);
  const sending = useRef(false);
  const firstShape = useShape(draft.choices[0].revisionId);
  const secondShape = useShape(draft.choices[1].revisionId);
  const sharedPresets = sharedPresetIds(firstShape.data, secondShape.data);
  const built = comparisonRequest(draft, sharedPresets);
  const { checked, checkComparison, acceptComparison, startComparison } = comparison;
  const stale = Boolean(checked && (!built.request || !sameComparisonRequest(built.request, checked.request)));

  useEffect(() => {
    if (checked) checkHeading.current?.focus();
  }, [checked]);
  useEffect(() => {
    if (comparison.experimentId) resultsHeading.current?.focus();
  }, [comparison.experimentId]);

  const updateChoice = (index: 0 | 1, next: ComparisonChoiceDraft) => {
    const choices: [ComparisonChoiceDraft, ComparisonChoiceDraft] = [...draft.choices];
    choices[index] = next;
    setDraft({ ...draft, choices });
  };

  const check = () => {
    if (sending.current || checkComparison.isPending) return;
    if (!built.request) {
      setProblems(built.problems);
      return;
    }
    setProblems([]);
    sending.current = true;
    checkComparison.mutate(built.request, { onSettled: () => { sending.current = false; } });
  };

  const accept = () => {
    if (!checked || stale || sending.current || acceptComparison.isPending) return;
    sending.current = true;
    acceptComparison.mutate(checked, {
      onSettled: () => { sending.current = false; },
      onError: (error) => { if (comparisonFailure(error).next === "check-again") comparison.clearCheck(); },
    });
  };

  const start = async (experiment: GenerationExperiment, confirmExpensive = false) => {
    if (sending.current || startComparison.isPending) return;
    if (!confirmExpensive && checked?.preflight.confirmation_required && checked.preflight.preflight_sha256 === experiment.preflight_sha256) {
      const yes = await confirm({ title: "Make two large pictures?", question: "These pictures are large and will take a while to make.",
        confirmLabel: "Make both pictures" });
      if (!yes) return;
      confirmExpensive = true;
    }
    sending.current = true;
    startComparison.mutate({ experiment, confirmExpensive }, {
      onSettled: () => { sending.current = false; },
      onError: async (error) => {
        if (comparisonFailure(error).next !== "confirm" || confirmExpensive) return;
        const yes = await confirm({ title: "Make two large pictures?", question: "These pictures are large and will take a while to make.",
          confirmLabel: "Make both pictures" });
        if (yes) void start(experiment, true);
      },
    });
  };

  const fresh = () => {
    comparison.forget();
    window.setTimeout(() => firstLabel.current?.focus(), 0);
  };

  if (comparison.experimentId) {
    return <div className="page-view generation-comparison">
      <header className="page-header"><div><h1>Compare generation choices</h1></div></header>
      <ComparisonResults experimentId={comparison.experimentId} onStart={(experiment) => void start(experiment)}
        starting={startComparison.isPending} startError={startComparison.error}
        onNew={fresh} onOpenChat={onOpenChat} headingRef={resultsHeading} />
      {confirmation}
    </div>;
  }

  const acceptFailure = acceptComparison.error ? comparisonFailure(acceptComparison.error) : null;
  return <div className="page-view generation-comparison">
    <header className="page-header"><div><h1>Compare generation choices</h1>
      <p>Two choices, one prompt: everything but the model and workflow is held the same.</p></div></header>
    <ComparisonSharedFields value={draft} onChange={setDraft} sharedPresets={sharedPresets}
      shapesKnown={firstShape.isSuccess && secondShape.isSuccess} />
    <div className="comparison-columns">
      <ComparisonChoiceFields legend="First choice" value={draft.choices[0]} labelRef={firstLabel}
        onChange={(next) => updateChoice(0, next)} />
      <ComparisonChoiceFields legend="Second choice" value={draft.choices[1]}
        onChange={(next) => updateChoice(1, next)} />
    </div>
    {problems.length > 0 && <ul className="comparison-problems" role="alert">{problems.map((problem) => <li key={problem}>{problem}</li>)}</ul>}
    <button type="button" className="primary" aria-disabled={checkComparison.isPending}
      onClick={check}>{checkComparison.isPending ? "Checking…" : "Check both choices"}</button>
    <ErrorCallout message={checkComparison.error ? comparisonFailure(checkComparison.error).message : null} />
    {acceptFailure && <><ErrorCallout message={acceptFailure.message} />
      <ComparisonRefusals refusals={acceptFailure.refusals} labels={draft.choices.map((choice) => choice.label.trim())} /></>}
    {checked && <ComparisonCheckResult checked={checked} stale={stale} accepting={acceptComparison.isPending}
      onAccept={accept} headingRef={checkHeading}
      onSeedPolicy={(kind) => setDraft({ ...draft, seed: { ...draft.seed, kind } })}
      onProfile={(ordinal, profileId) => updateChoice(ordinal === 1 ? 0 : 1, { ...draft.choices[ordinal === 1 ? 0 : 1], profileId })} />}
    {confirmation}
  </div>;
}
