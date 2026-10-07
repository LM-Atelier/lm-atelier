/** Server-resolved source canvas intent and preview; neither authorizes a turn. */

/** Extend keeps the whole source and paints new canvas around it; crop keeps
 * the centred part of the source that has the canvas's shape, and edits that. */
export type SourceFitMode = "extend" | "crop";

export interface SourceFitIntent {
  mode: SourceFitMode;
  width: number;
  height: number;
}

export interface SourceFitCapability {
  available: boolean;
  reason: "source_fit_workflow_unsupported" | null;
  modes: SourceFitMode[];
  request_authorized: false;
}

export interface SourceFitPreviewResult {
  version: 1;
  mode: SourceFitMode;
  workflow_revision_id: string;
  workflow_artifact_sha256: string;
  source_artifact_id: string;
  source: { width: number; height: number };
  canvas: { width: number; height: number };
  margins: { left: number; top: number; right: number; bottom: number };
  source_rectangle: { x: number; y: number; width: number; height: number };
  /** For a crop, the part of the source that fills the canvas, in the source's own pixels. */
  kept?: { left: number; top: number; width: number; height: number } | null;
  request_authorized: false;
}

/** The explicit source and revision whose canvas the person previewed. */
export interface SourceFitSelection {
  sourceArtifactId: string;
  workflowRevisionId: string;
  request: SourceFitIntent;
}

/** An explicit canvas supplies both dimensions without changing saved defaults. */
export function sourceCanvasSettings(settings: Record<string, unknown>, fit: SourceFitSelection | null | undefined): Record<string, unknown> {
  if (!fit) return settings;
  const next = { ...settings };
  delete next.width;
  delete next.height;
  return next;
}
