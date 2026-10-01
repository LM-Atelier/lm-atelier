// Browser mirrors of the generation comparison contracts, with the server's field names.
import type { OutputRatioPresetId, WorkStepStatus } from "./types";

export type SeedPolicyKind = "fixed_numeric" | "same_recorded_number" | "independent_deterministic" | "random_per_trial";

export type GenerationExperimentRefusalCode =
  | "arm-profile-unavailable" | "arm-workflow-unavailable" | "arm-operation-mismatch" | "arm-workflow-untrusted"
  | "arm-activation-not-ready" | "arm-package-missing" | "arm-model-mismatch" | "arm-engine-unavailable"
  | "arm-setting-unsupported" | "arm-setting-invalid" | "common-input-overridden" | "arm-input-unsupported"
  | "arm-seed-unsupported" | "arm-geometry-unreachable" | "arm-lora-refused" | "seed-family-unproven"
  | "arms-identical" | "experiment-too-large" | "arm-changed";

export type TrialWorkStatus = WorkStepStatus | "removed";

export interface PresetGeometry {
  mode: "preset";
  preset_id: OutputRatioPresetId;
}

export interface SizeGeometry {
  mode: "size";
  width: number;
  height: number;
}

export type ExperimentGeometry = PresetGeometry | SizeGeometry;

export interface SeedPolicy {
  kind: SeedPolicyKind;
  seed: number | null;
}

export interface ExperimentArmRequest {
  label: string;
  profile_id: string;
  workflow_revision_id: string;
  settings: Record<string, unknown>;
}

export interface GenerationExperimentRequest {
  name: string;
  operation: "text_to_image";
  prompt: string;
  negative_prompt: string;
  geometry: ExperimentGeometry;
  seed_policy: SeedPolicy;
  arms: ExperimentArmRequest[];
  /** Blind hides which picture each choice made until the preference is said; left out, the choices are named. */
  evaluation_mode?: "blind";
}

export interface GenerationExperimentCreate extends GenerationExperimentRequest {
  idempotency_key: string;
  preflight_sha256: string;
}

export interface GenerationExperimentStart {
  idempotency_key: string;
  snapshot_sha256: string;
  confirm_expensive: boolean;
}

export interface RefusalAlternative {
  seed_policy: SeedPolicyKind | null;
  profile_id: string | null;
}

export interface ExperimentRefusal {
  code: GenerationExperimentRefusalCode;
  arm_ordinal: number | null;
  setting: string | null;
  alternative: RefusalAlternative | null;
  message: string;
}

export interface ResourceEvidence {
  resource: "work_units" | "output_bytes";
  value: number;
  unit: "work_units" | "bytes";
  kind: "measured" | "estimated";
  source: "admission_formula";
  confidence: "heuristic";
}

export interface ArmPreflight {
  ordinal: number;
  label: string;
  outcome: "compatible" | "refused";
  profile_id: string;
  profile_name: string | null;
  workflow_revision_id: string;
  workflow_version: number | null;
  workflow_activation_id: string | null;
  model_family: string | null;
  width: number | null;
  height: number | null;
  effective_settings: Record<string, unknown>;
  trigger_words_applied: string[];
  snapshot_sha256: string | null;
}

export interface GenerationExperimentPreflight {
  outcome: "compatible" | "refused";
  preflight_sha256: string | null;
  seed_equivalence: "same_family" | "none";
  confirmation_required: boolean;
  estimate: ResourceEvidence[];
  arms: ArmPreflight[];
  refusals: ExperimentRefusal[];
}

export interface ExperimentTrial {
  id: string;
  ordinal: number;
  seed: number;
  state: "planned" | "started";
  work_step_id: string | null;
  run_id: string | null;
  job_id: string | null;
  status: TrialWorkStatus | null;
}

export interface ExperimentArm {
  id: string;
  ordinal: number;
  label: string;
  profile_id: string;
  profile_name: string | null;
  workflow_revision_id: string;
  workflow_version: number | null;
  workflow_activation_id: string | null;
  model_family: string | null;
  width: number;
  height: number;
  effective_settings: Record<string, unknown>;
  trigger_words_applied: string[];
  snapshot_sha256: string;
  trials: ExperimentTrial[];
}

/** What the person said of the two pictures: their own judgement, kept as said. */
export type GenerationExperimentPreference = "preferred" | "tied" | "unsuitable";

export interface ExperimentEvaluation {
  preference: GenerationExperimentPreference;
  mode: "unblinded" | "blind";
  arm_ordinal: number | null;
  note: string | null;
  created_at: string;
}

/** One saying: a choice preferred by its ordinal, or a tie, or neither suiting. */
export type GenerationExperimentEvaluationCreate =
  | { preference: "preferred"; arm_ordinal: number; note?: string }
  | { preference: "tied" | "unsuitable"; note?: string };

export interface GenerationExperiment {
  id: string;
  name: string;
  state: "ready" | "started";
  operation: "text_to_image";
  prompt: string;
  negative_prompt: string;
  geometry: ExperimentGeometry;
  seed_policy: SeedPolicy;
  seed_equivalence: "same_family" | "none";
  preflight_sha256: string;
  snapshot_sha256: string;
  estimate: ResourceEvidence[];
  created_at: string;
  work_plan_id: string | null;
  started_at: string | null;
  arms: ExperimentArm[];
  /** The latest thing said of the pictures; null until something is. */
  evaluation: ExperimentEvaluation | null;
  evaluation_mode: "unblinded" | "blind";
  /** True while a blind comparison waits for its blind saying: no trial is linked to its picture. */
  blind_pending: boolean;
}

/** A setting a choice ran with that the recipe drafted from it does not hold, and why. */
export interface RecipeDraftLeftOut {
  setting: string;
  reason: string;
  message: string;
}

/** A recipe to review before saving: one choice's settings, as a recipe holds them, beside the model and workflow it ran on. */
export interface GenerationExperimentRecipeDraft {
  experiment_id: string;
  arm_ordinal: number;
  use_case: "image_generation";
  name: string;
  settings_json: Record<string, unknown>;
  left_out: RecipeDraftLeftOut[];
  profile_id: string;
  profile_name: string | null;
  workflow_id: string;
  workflow_family_id: string | null;
  workflow_revision_id: string;
  workflow_name: string;
  workflow_version: number | null;
}

/** One picture in a blind viewing, known only by where it is shown. */
export interface BlindPosition {
  position: number;
  status: TrialWorkStatus | null;
  ready: boolean;
}

export interface BlindReveal {
  position: number;
  arm_ordinal: number;
  label: string;
}

/** A blind comparison as one viewing shows it: its own order, no choice named until the saying. */
export interface GenerationExperimentBlindView {
  id: string;
  experiment_id: string;
  positions: BlindPosition[];
  evaluation: ExperimentEvaluation | null;
  reveal: BlindReveal[] | null;
}

/** A preference said in a blind viewing: a picture by its position, or a tie, or neither. */
export type GenerationExperimentBlindEvaluationCreate =
  | { preference: "preferred"; position: number; note?: string }
  | { preference: "tied" | "unsuitable"; note?: string };
