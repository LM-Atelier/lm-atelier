import type { SettingField } from "./types";

/** The server's answer for one picture, chat and workflow choice. */
export interface EnlargementPreview {
  version: 1;
  status: "ready";
  /** The workflow an Enhance of this picture runs; Apply names it, so the run is the one previewed. */
  workflow_revision_id: string;
  /** What the person can choose, when the workflow applies a chosen factor. */
  factor: SettingField | null;
  /** How much a workflow that always enlarges by the same amount enlarges, when its graph proves it. */
  fixed_factor: number | null;
  request_authorized: false;
}
