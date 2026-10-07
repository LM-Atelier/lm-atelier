import type { Dispatch, SetStateAction } from "react";
import type { SendFromComposer } from "./chatComposerContracts";
import type { composerSubmission } from "./composerSubmission";
import { sourceCanvasSettings, type SourceFitSelection } from "./sourceFit";
import { buildTurnRequest } from "./turnRequest";
import type { RoutingMode, WorkflowFamily, WorkflowSelection } from "./types";
import type { ComposerAttachment } from "./useComposerUploads";
import { useSourceFitCanvas, type SourceFitPreviewContext } from "./useSourceFitCanvas";
import type { TurnEditorState } from "./useTurnEditorState";

export const SOURCE_FIT_PREVIEW_REQUIRED = "Preview the selected source canvas before sending.";

/** The canvas a composer's first picture can be fitted to, and what a send may carry from it.

A chosen canvas is only sent once its preview is current, and only in a mode that
can draw it; the settings it would override are dropped from the send so the two
cannot disagree about the output size. The picture the preview was fitted to goes
first among the send's inputs, so a picture the server chose from the chat is the
one the turn is admitted with. */
export function useTurnEditorSourceFit({
  mode, attachments, value, families, sourceCanvasRevisionId, workflowSelection, selections, projectSelections,
  previewContext, updateState, setAcceptanceError,
}: {
  mode: RoutingMode;
  attachments: ComposerAttachment[];
  value: SourceFitSelection | null | undefined;
  families: WorkflowFamily[] | undefined;
  sourceCanvasRevisionId?: string | null;
  workflowSelection: WorkflowSelection | null | undefined;
  selections: WorkflowSelection[] | undefined;
  projectSelections: WorkflowSelection[] | null | undefined;
  /** The submission a preview is fitted within; without one the preview asks only about the workflow. */
  previewContext: SourceFitPreviewContext | undefined;
  updateState: (update: SetStateAction<TurnEditorState>) => void;
  setAcceptanceError: Dispatch<SetStateAction<string>>;
}) {
  const primarySourceId = attachments[0]?.kind === "image" ? attachments[0].id : null;
  const canvas = useSourceFitCanvas({
    mode, sourceId: primarySourceId, value, families: families ?? [], sourceCanvasRevisionId, previewContext,
    workflowSelection: workflowSelection ?? selections?.find((one) => one.selector_capability === "image"),
    projectSelection: projectSelections?.find((one) => one.selector_capability === "image") ?? null,
    onChange: (sourceFit) => {
      updateState((current) => ({ ...current, sourceFit }));
      setAcceptanceError("");
    },
  });
  const shown = Boolean(((mode === "image" || mode === "auto") && primarySourceId) || value);
  const forSend = (selectedMode: RoutingMode, settings: Record<string, unknown>, inputArtifactIds: string[]) => {
    const sourceFit = value ? canvas.selection : value;
    if (value && (!sourceFit || (selectedMode !== "image" && selectedMode !== "auto"))) return null;
    const source = sourceFit?.sourceArtifactId;
    return {
      sourceFit,
      settings: sourceCanvasSettings(settings, sourceFit),
      inputArtifactIds: source && inputArtifactIds[0] !== source
        ? [source, ...inputArtifactIds.filter((id) => id !== source)] : inputArtifactIds,
    };
  };
  return { primarySourceId, canvas, shown, forSend };
}

/** The ordinary send a canvas would be previewed within, built as the send itself builds it. */
export function turnPreviewContext(
  chatId: string,
  text: string,
  mode: RoutingMode,
  attachments: ComposerAttachment[],
  submission: ReturnType<typeof composerSubmission>,
  value: SourceFitSelection | null | undefined,
): SourceFitPreviewContext | undefined {
  if (mode !== "image" && mode !== "auto") return undefined;
  return {
    kind: "turn",
    id: chatId,
    request: buildTurnRequest({
      text: text.trim(),
      mode,
      inputArtifactIds: attachments.map((item) => item.id),
      settings: sourceCanvasSettings(submission.settings, value),
      references: submission.references,
      outputCount: submission.requestedOutputCount,
      promptSource: submission.promptSource,
    }),
  };
}

/** Send with the fitted canvas only when there is one, so a send without it keeps its old shape. */
export function sendWithSourceFit(
  send: SendFromComposer,
  [text, mode, artifacts, settings, references, outputCount, promptSource]: Parameters<SendFromComposer>,
  sourceFit: SourceFitSelection | null | undefined,
) {
  if (sourceFit) send(text, mode, artifacts, settings, references, outputCount, promptSource, sourceFit);
  else send(text, mode, artifacts, settings, references, outputCount, promptSource);
}
