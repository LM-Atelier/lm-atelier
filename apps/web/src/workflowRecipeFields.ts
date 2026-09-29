import { resolveCapabilitySettings, resolveWorkflowSettings } from "./settings";
import type { EngineCapabilities, EngineRole, SettingField } from "./types";
import type { RecipeSettingValue, WorkflowUseCase } from "./workflowUseCaseTypes";

export function recipeOperation(useCase: WorkflowUseCase): string {
  if (useCase === "chat") return "text";
  if (useCase === "image_generation") return "text_to_image";
  if (useCase === "video_generation") return "text_to_video";
  if (useCase === "video_animate") return "image_to_video";
  return "image_to_image";
}

export function recipeRole(useCase: WorkflowUseCase): EngineRole {
  return useCase === "chat" ? "chat" : useCase.startsWith("video_") ? "video" : "image";
}

export function isRecipeSettingValue(value: unknown): value is RecipeSettingValue {
  if (value === null || typeof value === "boolean" || typeof value === "string") return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(isRecipeSettingValue);
  return typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype
    && Object.values(value).every(isRecipeSettingValue);
}

export function recipeFields(engine: EngineCapabilities, useCase: WorkflowUseCase, schema: Record<string, unknown>): SettingField[] {
  const properties = schema.properties && typeof schema.properties === "object" ? schema.properties : {};
  return resolveWorkflowSettings(resolveCapabilitySettings(engine, recipeRole(useCase)), schema).filter((field) => {
    const property: unknown = Reflect.get(properties, field.key);
    const readOnly = property !== null && typeof property === "object" && Reflect.get(property, "readOnly") === true;
    return field.key !== "prompt" && field.key !== "negative_prompt" && field.available
      && field.scope !== "load" && !readOnly && isRecipeSettingValue(field.default);
  });
}

export function recipeValueError(field: SettingField, value: RecipeSettingValue): string | null {
  const invalid = `Choose a valid value for ${field.label}.`;
  if (field.choices.length && !field.choices.some((choice) => JSON.stringify(choice) === JSON.stringify(value))) return invalid;
  if (field.type === "integer" || field.type === "number") {
    if (typeof value !== "number" || !Number.isFinite(value) || (field.type === "integer" && !Number.isInteger(value))) return invalid;
    if ((field.minimum !== null && value < field.minimum) || (field.maximum !== null && value > field.maximum)) return invalid;
    if (field.multiple_of != null && Math.abs(value / field.multiple_of - Math.round(value / field.multiple_of)) > 1e-8) return invalid;
  } else if (field.type === "boolean" && typeof value !== "boolean") return invalid;
  else if (field.type === "string" && typeof value !== "string") return invalid;
  else if (field.type === "array" && !Array.isArray(value)) return invalid;
  else if (field.type === "object" && (value === null || typeof value !== "object" || Array.isArray(value))) return invalid;
  return null;
}
