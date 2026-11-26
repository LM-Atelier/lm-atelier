// Browser mirror of what an accepted render changed on purpose, with the server's field names.
import type { SettingSource } from "./settingRefusalTypes";

export type SettingChangeCode =
  | "set-aside-not-taken" | "set-aside-not-offered" | "set-aside-fixed" | "set-aside-not-usable"
  | "size-follows-source" | "set-aside-for-offer" | "outputs-from-batch-size"
  | "batch-size-for-one-output" | "output-count-raised" | "one-output"
  | "strength-default-for-new-picture" | "strength-full-for-extension"
  | "length-aligned-to-frames" | "seed-drawn-again" | "seed-per-output" | "seed-per-draft";

/** One value an accepted render changed on purpose, and why; never a value it may not name. */
export interface SettingChange {
  code: SettingChangeCode;
  /**
   * The setting's name, or the request field that is no setting, such as "output_count". "*" is a
   * setting whose key is not a plain setting name, or more settings than one item names.
   */
  field: string;
  message: string;
  /** Absent when this request sent the value itself. */
  source?: SettingSource;
  /** The value it was given, when that is a number, true or false, or one of its own choices. */
  before?: number | boolean | string;
  /** The value used instead, named by the same rule. */
  after?: number | boolean | string;
  /** The step of the plan the change was made for, counting from 1, when not every step was. */
  step?: number;
}
