// Browser mirror of why a setting was refused, with the server's field names.

export type SettingRefusalCode =
  | "setting-unknown" | "setting-not-offered" | "setting-not-an-integer" | "setting-not-a-number"
  | "setting-not-a-string" | "setting-not-a-boolean" | "setting-not-an-array" | "setting-not-an-object"
  | "setting-not-a-choice" | "setting-below-minimum" | "setting-above-maximum" | "setting-not-a-multiple"
  | "setting-not-finite" | "setting-too-long" | "setting-too-many-items" | "setting-too-deep"
  | "setting-too-complex" | "settings-too-many" | "setting-not-applicable" | "setting-needs-its-pair"
  | "setting-needs-a-role" | "setting-role-unknown" | "setting-not-taken-by-every-workflow"
  | "plan-too-many-outputs" | "plan-too-few-outputs"
  | "plan-too-many-pending" | "plan-too-much-work" | "plan-storage-too-large" | "plan-storage-unavailable"
  | "plan-video-too-long" | "plan-too-many-steps" | "plan-prompt-too-long" | "plan-step-prompt-too-long"
  | "plan-one-output-per-step";

/** Where a refused value came from when the request did not send it. */
export type SettingSource =
  | "inherited" | "turn_preset" | "chat" | "chat_preset" | "project" | "project_preset"
  | "default_preset" | "profile" | "saved" | "planned";

/** One setting a request was refused for, and why; never the value that was refused. */
export interface SettingError {
  code: SettingRefusalCode;
  /**
   * The setting's name, "*" for one whose key is not a plain setting name, or the part of the
   * request that is no setting, such as "plan" for a limit of the whole plan.
   */
  field: string;
  message: string;
  /** The bound the value passed, for a code that has one; for a plan, its fixed limit. */
  limit?: number;
  /** The first choices the setting takes, when each one shown is a plain value. */
  allowed?: (string | number | boolean)[];
  /** Absent when this request sent the value itself. */
  source?: SettingSource;
  /** The step of an ordered plan the refusal is about, counting from 1. */
  step?: number;
}
