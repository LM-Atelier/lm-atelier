/** Which settings size what a workflow revision makes; see `WorkflowSize`. */
export type WorkflowSizeControl =
  | "width_height"
  | "aspect_ratio"
  | "aspect_ratio_and_width_height"
  | "workflow";

/**
 * One side of a size, under the key a turn's settings name it by.
 *
 * `multiple_of` is enforced; `step` is only the increment the workflow's own
 * control moves by, but a value off it may be rounded by the workflow.
 * `choices`, when there are any, are the only values accepted. The default and
 * bounds of a whole-number side are integers, so any of them can be sent back.
 */
export interface WorkflowSizeSide {
  key: string;
  default: number | string | null;
  minimum: number | null;
  maximum: number | null;
  multiple_of: number | null;
  step: number | null;
  choices: (number | string)[];
}

/** A size chosen from a menu, such as an aspect ratio; only the choices are accepted. */
export interface WorkflowSizeChoice {
  key: string;
  default: number | string | null;
  choices: (number | string)[];
}

/** A number that scales a chosen shape, such as a pixel budget. */
export interface WorkflowSizeNumber {
  key: string;
  default: number | null;
  minimum: number | null;
  maximum: number | null;
  multiple_of: number | null;
  step: number | null;
}

/**
 * Which settings size what one workflow revision makes, as a turn checks them.
 *
 * Each side is described on its own: a side the workflow fixes is null, and a
 * turn sends only the sides that are present. `workflow` means no single
 * control sizes the result, so a turn sends none. Nothing is snapped or
 * clamped: a value the control does not offer is refused.
 */
export interface WorkflowSize {
  version: 1;
  control: WorkflowSizeControl;
  width: WorkflowSizeSide | null;
  height: WorkflowSizeSide | null;
  aspect_ratio: WorkflowSizeChoice | null;
  megapixels: WorkflowSizeNumber | null;
  round_to_multiple: WorkflowSizeNumber | null;
}
