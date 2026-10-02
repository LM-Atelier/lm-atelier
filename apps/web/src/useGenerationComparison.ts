import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api } from "./api";
import type {
  GenerationExperiment,
  GenerationExperimentPreflight,
  GenerationExperimentRequest,
} from "./generationExperimentTypes";

const REMEMBERED = "lm-atelier.generation-comparison";

export interface CheckedComparison {
  request: GenerationExperimentRequest;
  preflight: GenerationExperimentPreflight;
  createKey: string;
}

function newKey(): string {
  return typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function readRemembered(): string | null {
  try {
    return window.sessionStorage.getItem(REMEMBERED);
  } catch {
    return null;
  }
}

function writeRemembered(experimentId: string | null): void {
  try {
    if (experimentId) window.sessionStorage.setItem(REMEMBERED, experimentId);
    else window.sessionStorage.removeItem(REMEMBERED);
  } catch {
    // A tab that cannot keep it simply forgets the comparison on reload.
  }
}

/** Check, accept and start one comparison; each retry reuses the key of the request it repeats. */
export function useGenerationComparison() {
  const client = useQueryClient();
  const [checked, setChecked] = useState<CheckedComparison | null>(null);
  const [experimentId, setExperimentIdState] = useState<string | null>(readRemembered);
  const latestCheck = useRef(0);
  const startKeys = useRef(new Map<string, string>());

  const setExperimentId = (next: string | null) => {
    writeRemembered(next);
    setExperimentIdState(next);
  };

  const checkComparison = useMutation({
    mutationKey: ["generation-experiments", "check"],
    mutationFn: async (request: GenerationExperimentRequest) => {
      const sequence = ++latestCheck.current;
      const preflight = await api.preflightGenerationExperiment(request);
      return { sequence, request, preflight };
    },
    onSuccess: ({ sequence, request, preflight }) => {
      // An answer to an older check, or one that does not describe these two choices, is not kept.
      if (sequence !== latestCheck.current) return;
      if (preflight.arms.length !== 2 || preflight.arms.some((arm, index) => arm.label !== request.arms[index]?.label)) return;
      setChecked({ request, preflight, createKey: newKey() });
    },
  });

  const acceptComparison = useMutation({
    mutationKey: ["generation-experiments", "accept"],
    mutationFn: (current: CheckedComparison) => {
      if (!current.preflight.preflight_sha256) throw new Error("Only a comparison whose check passed can be accepted.");
      return api.createGenerationExperiment({
        ...current.request,
        idempotency_key: current.createKey,
        preflight_sha256: current.preflight.preflight_sha256,
      });
    },
    onSuccess: (experiment: GenerationExperiment, current) => {
      if (experiment.preflight_sha256 !== current.preflight.preflight_sha256) return;
      client.setQueryData(["generation-experiments", experiment.id], experiment);
      setExperimentId(experiment.id);
    },
  });

  const startComparison = useMutation({
    mutationKey: ["generation-experiments", "start"],
    mutationFn: ({ experiment, confirmExpensive }: { experiment: GenerationExperiment; confirmExpensive: boolean }) => {
      const frozen = `${experiment.id}:${experiment.snapshot_sha256}:${confirmExpensive}`;
      const key = startKeys.current.get(frozen) ?? newKey();
      startKeys.current.set(frozen, key);
      return api.startGenerationExperiment(experiment.id, {
        idempotency_key: key, snapshot_sha256: experiment.snapshot_sha256, confirm_expensive: confirmExpensive,
      });
    },
    onSuccess: (experiment: GenerationExperiment, { experiment: asked }) => {
      if (experiment.id !== asked.id || experiment.state !== "started") return;
      if (experiment.arms.some((arm) => arm.trials.some((trial) => !trial.run_id))) return;
      client.setQueryData(["generation-experiments", experiment.id], experiment);
    },
  });

  return {
    checked,
    experimentId,
    checkComparison,
    acceptComparison,
    startComparison,
    clearCheck: () => setChecked(null),
    forget: () => {
      setChecked(null);
      setExperimentId(null);
      checkComparison.reset();
      acceptComparison.reset();
      startComparison.reset();
    },
  };
}
