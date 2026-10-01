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
}
