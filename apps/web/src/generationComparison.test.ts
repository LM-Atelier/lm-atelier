import { describe, expect, it } from "vitest";
import {
  comparisonFailure,
  comparisonIsWorking,
  comparisonRequest,
  EMPTY_COMPARISON,
  keptPicture,
  refusalSubject,
  sameComparisonRequest,
  settingRows,
  sharedPresetIds,
  type ComparisonDraft,
} from "./generationComparison";
import type { ExperimentTrial, GenerationExperiment } from "./generationExperimentTypes";
import type { Run } from "./types";

function draft(overrides: Partial<ComparisonDraft> = {}): ComparisonDraft {
  return {
    ...EMPTY_COMPARISON,
    prompt: "A quiet harbor at dawn",
    choices: [
      { label: " Fewer steps ", profileId: "profile-a", revisionId: "revision-a" },
      { label: "More steps", profileId: "profile-b", revisionId: "revision-b" },
    ],
    ...overrides,
  };
}

describe("comparisonRequest", () => {
  it("builds the exact request a draft describes", () => {
    const built = comparisonRequest(draft({ seed: { kind: "same_recorded_number", number: "41" } }), []);
    expect(built.problems).toEqual([]);
    expect(built.request).toEqual({
      name: "Fewer steps and More steps",
      operation: "text_to_image",
      prompt: "A quiet harbor at dawn",
      negative_prompt: "",
      geometry: { mode: "size", width: 1024, height: 1024 },
      seed_policy: { kind: "same_recorded_number", seed: 41 },
      arms: [
        { label: "Fewer steps", profile_id: "profile-a", workflow_revision_id: "revision-a", settings: {} },
        { label: "More steps", profile_id: "profile-b", workflow_revision_id: "revision-b", settings: {} },
      ],
    });
  });

  it("asks for a blind comparison only when one was chosen", () => {
    const blind = comparisonRequest(draft({ blind: true }), []).request;
    const named = comparisonRequest(draft(), []).request;
    expect(blind).toMatchObject({ evaluation_mode: "blind" });
    // Left out rather than sent as the default, so the request is exactly what it always was.
    expect(named && "evaluation_mode" in named).toBe(false);
    expect(blind && named && sameComparisonRequest(blind, named)).toBe(false);
  });

  it("sends no seed for a random seed per picture, even when one was typed", () => {
    const built = comparisonRequest(draft({ seed: { kind: "random_per_trial", number: "7" } }), []);
    expect(built.request?.seed_policy).toEqual({ kind: "random_per_trial", seed: null });
  });

  it.each([
    ["labels that fold to the same text", draft({ choices: [
      { label: "Straße", profileId: "a", revisionId: "a" }, { label: "STRASSE", profileId: "b", revisionId: "b" },
    ] }), "Give each choice its own label."],
    ["a missing model", draft({ choices: [
      { label: "One", profileId: "", revisionId: "a" }, { label: "Two", profileId: "b", revisionId: "b" },
    ] }), "Choose an image model for each choice."],
    ["a blank prompt", draft({ prompt: "   " }), "Write the prompt both choices share."],
    ["a size that is not a whole number", draft({ size: { mode: "size", width: "10.5", height: "20" } }), "Enter a whole-number width and height."],
    ["a fixed seed without its number", draft({ seed: { kind: "fixed_numeric", number: "" } }), "Enter the seed both choices start from."],
    ["a seed past the largest", draft({ seed: { kind: "same_recorded_number", number: "2147483648" } }), "Enter a seed from 0 to 2147483647."],
    ["a shape neither workflow is shown to make", draft({ size: { mode: "preset", presetId: "16:9" } }), "Choose a shape both workflows can make."],
  ])("refuses %s before anything is sent", (_case, value, problem) => {
    const built = comparisonRequest(value, ["1:1"]);
    expect(built.request).toBeNull();
    expect(built.problems).toContain(problem);
  });

  it("accepts a shape both workflows can make", () => {
    const built = comparisonRequest(draft({ size: { mode: "preset", presetId: "1:1" } }), ["1:1"]);
    expect(built.request?.geometry).toEqual({ mode: "preset", preset_id: "1:1" });
  });
});

describe("helpers", () => {
  it("compares requests whatever order their keys are in", () => {
    const built = comparisonRequest(draft(), []).request;
    expect(built).not.toBeNull();
    const reordered = Object.fromEntries(Object.entries(built!).reverse()) as typeof built;
    expect(Object.keys(reordered!)[0]).toBe("arms");
    expect(sameComparisonRequest(built!, reordered!)).toBe(true);
    expect(sameComparisonRequest(built!, { ...built!, prompt: "Something else" })).toBe(false);
  });

  it("offers only shapes both workflows can be shown to make", () => {
    expect(sharedPresetIds({ available: true, preset_ids: ["1:1", "16:9"] }, { available: true, preset_ids: ["16:9", "3:2"] }))
      .toEqual(["16:9"]);
    expect(sharedPresetIds({ available: false, preset_ids: ["1:1"] }, { available: true, preset_ids: ["1:1"] })).toEqual([]);
  });

  it("lists every setting side by side, marks differences and leaves out the shared negative prompt", () => {
    expect(settingRows(
      { effective_settings: { steps: 8, cfg: 4, negative_prompt: "blurry" } },
      { effective_settings: { steps: 20, cfg: 4, negative_prompt: "blurry", sampler: "euler" } },
    )).toEqual([
      { key: "cfg", values: ["4", "4"], differs: false },
      { key: "sampler", values: ["not set", "euler"], differs: true },
      { key: "steps", values: ["8", "20"], differs: true },
    ]);
  });

  it("names the choice a refusal is about, or the whole comparison", () => {
    expect(refusalSubject({ arm_ordinal: 2 }, ["Left", "Right"])).toBe("Right");
    expect(refusalSubject({ arm_ordinal: null }, ["Left", "Right"])).toBe("Whole comparison");
  });

  it("keeps working only while a started picture is still waiting or being made", () => {
    const experiment = (statuses: ExperimentTrial["status"][]) => ({
      state: "started",
      arms: statuses.map((status) => ({ trials: [{ status }] })),
    }) as unknown as GenerationExperiment;
    expect(comparisonIsWorking(experiment(["running", "complete"]))).toBe(true);
    expect(comparisonIsWorking(experiment(["failed", "complete"]))).toBe(false);
    expect(comparisonIsWorking(undefined)).toBe(false);
  });
});

describe("keptPicture", () => {
  const trial: ExperimentTrial = {
    id: "trial-1", ordinal: 1, seed: 4, state: "started", work_step_id: "step-1", run_id: "run-1", job_id: "job-1", status: "complete",
  };
  const run = (outputs: unknown[], trialId = "trial-1") => ({
    id: "run-1", work_step_id: "step-1", provenance_json: { generation_experiment: { trial_id: trialId }, outputs },
  }) as unknown as Run;

  it("returns the kept picture and passes over a preview the engine does not keep", () => {
    expect(keptPicture(run([
      { artifact_id: "preview", kind: "image", output_origin: { state: "attributed", output_type: "temp" } },
      { artifact_id: "kept", kind: "image", output_origin: { state: "attributed", output_type: "output" } },
    ]), trial)).toBe("kept");
  });

  it("shows nothing from a run that is not this picture's", () => {
    expect(keptPicture(run([{ artifact_id: "kept", kind: "image" }], "trial-2"), trial)).toBeNull();
  });
});

describe("comparisonFailure", () => {
  it("reads refusals and the estimate from the error body", () => {
    const error = Object.assign(new Error("A choice in this comparison cannot run as asked."), {
      status: 422,
      code: "generation-experiment-refused",
      payload: {
        refusals: [{ code: "arm-profile-unavailable", arm_ordinal: 2, setting: null, alternative: null, message: "Not available." }, { nonsense: true }],
        estimate: [{ resource: "work_units", value: 9, unit: "work_units", kind: "estimated", source: "admission_formula", confidence: "heuristic" }],
      },
    });
    const failure = comparisonFailure(error);
    expect(failure.next).toBe("check-again");
    expect(failure.refusals.map((refusal) => refusal.code)).toEqual(["arm-profile-unavailable"]);
    expect(failure.estimate.map((item) => item.value)).toEqual([9]);
  });

  it.each([
    ["generation-experiment-confirmation-required", "confirm"],
    ["generation-experiment-snapshot-changed", "read-again"],
    ["generation-experiment-not-found", "gone"],
    ["generation-experiment-unavailable", "retry"],
  ])("maps %s to %s", (code, next) => {
    expect(comparisonFailure(Object.assign(new Error("x"), { code })).next).toBe(next);
  });
});
