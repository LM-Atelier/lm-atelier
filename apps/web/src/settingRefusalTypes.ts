// Browser mirror of why a setting was refused, with the server's field names.

export type SettingRefusalCode =
  | "setting-unknown" | "setting-not-offered" | "setting-not-an-integer" | "setting-not-a-number"
  | "setting-not-a-string" | "setting-not-a-boolean" | "setting-not-an-array" | "setting-not-an-object"
  | "setting-not-a-choice" | "setting-below-minimum" | "setting-above-maximum" | "setting-not-a-multiple"
  | "setting-not-finite" | "setting-too-long" | "setting-too-many-items" | "setting-too-deep"
  | "setting-too-complex" | "settings-too-many" | "setting-not-applicable" | "setting-needs-its-pair";

/** Where a refused value came from when the request did not send it. */
export type SettingSource =
  | "inherited" | "turn_preset" | "chat" | "chat_preset" | "project" | "project_preset"
  | "default_preset" | "profile" | "saved";

/** One setting a request was refused for, and why; never the value that was refused. */
export interface SettingError {
  code: SettingRefusalCode;
  /** The setting's name, or "*" for one whose key is not a plain setting name. */
  field: string;
  message: string;
  /** The bound the value passed, for a code that has one. */
  limit?: number;
  /** The first choices the setting takes, when each one shown is a plain value. */
  allowed?: (string | number | boolean)[];
  /** Absent when this request sent the value itself. */
  source?: SettingSource;
}
