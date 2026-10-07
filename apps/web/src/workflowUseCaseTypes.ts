export const workflowUseCases = [
  { id: "chat", label: "Chat" },
  { id: "image_generation", label: "Image generation" },
  { id: "image_edit", label: "Whole-image edit" },
  { id: "image_inpaint", label: "Masked inpaint" },
  { id: "image_outpaint", label: "Outpaint" },
  { id: "image_upscale", label: "Image upscale" },
  { id: "video_generation", label: "Video generation" },
  { id: "video_animate", label: "Image animation" },
] as const;

export type WorkflowUseCase = typeof workflowUseCases[number]["id"];
export type RecipeSettingValue = null | boolean | number | string | RecipeSettingValue[]
  | { [key: string]: RecipeSettingValue };

export interface WorkflowUseCasePresetCreate {
  name: string;
  use_case: WorkflowUseCase;
  settings_json: Record<string, RecipeSettingValue>;
  enabled: boolean;
  is_default: boolean;
}
export interface WorkflowUseCasePreset extends WorkflowUseCasePresetCreate {
  id: string;
  builtin: boolean;
}
export type WorkflowUseCaseChoice = { mode: "inherit" } | { mode: "automatic" }
  | { mode: "preset"; preset_id: string };
export type WorkflowUseCaseDefault = { preset_id: string | null };
export type WorkflowRecipeTarget = { kind: "chat" | "project"; id: string };
export type WorkflowRecipeScope = WorkflowRecipeTarget | { kind: "workspace" };
