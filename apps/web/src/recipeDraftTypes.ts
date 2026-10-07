import type { RecipeDraftLeftOut } from "./generationExperimentTypes";

/** What every recipe draft holds, whatever it was drafted from: settings only, beside the model and workflow they ran with. */
export interface RecipeDraft {
  use_case: "image_generation" | "video_generation" | "video_animate";
  name: string;
  settings_json: Record<string, unknown>;
  left_out: RecipeDraftLeftOut[];
  profile_id: string | null;
  profile_name: string | null;
  workflow_id: string;
  workflow_family_id: string | null;
  workflow_name: string;
  workflow_version: number | null;
}

/** The words an edit was asked with, to start an Image Studio recipe from it. */
export interface EditRecipeDraft {
  run_id: string;
  instruction: string;
}

/** A recipe to review before saving: the settings one finished generation ran with, as a recipe holds them. */
export interface OutputRecipeDraft {
  run_id: string;
  use_case: "image_generation" | "video_generation" | "video_animate";
  name: string;
  settings_json: Record<string, unknown>;
  left_out: RecipeDraftLeftOut[];
  profile_id: string | null;
  profile_name: string | null;
  workflow_id: string;
  workflow_family_id: string | null;
  workflow_revision_id: string;
  workflow_name: string;
  workflow_version: number;
}
