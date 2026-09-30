import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { workflowRevisionForTurn } from "./turnEditorContext";
import type { PriorTurnEditRequest, RoutingMode, WorkflowFamily, WorkflowSelection } from "./types";
import type { SourceFitIntent, SourceFitMode, SourceFitSelection } from "./sourceFit";
import type { TurnRequestPayload } from "./turnRequest";

export type SourceFitPreviewContext =
  | { kind: "turn"; id: string; request: TurnRequestPayload }
  | { kind: "prior-edit"; id: string; request: PriorTurnEditRequest };

export function sourceCanvasDimension(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 && value <= 1_000_000;
}

/** Preview is requested explicitly and belongs to one complete submission context. */
export function useSourceFitCanvas({
  mode, families, workflowSelection, projectSelection, sourceId, value, onChange, sourceCanvasRevisionId, previewContext,
}: {
  mode: RoutingMode;
  sourceCanvasRevisionId?: string | null;
  families: WorkflowFamily[];
  workflowSelection: WorkflowSelection | null | undefined;
  projectSelection: WorkflowSelection | null | undefined;
  sourceId: string | null;
  value: SourceFitSelection | null | undefined;
  onChange: (value: SourceFitSelection | null) => void;
  previewContext?: SourceFitPreviewContext;
}) {
  const revisionId = mode === "image" || mode === "auto"
    ? sourceCanvasRevisionId !== undefined ? sourceCanvasRevisionId
      : workflowRevisionForTurn("image", Boolean(sourceId), families, workflowSelection, projectSelection)
    : null;
  const [requestedKey, setRequestedKey] = useState<string | null>(null);
  const capability = useQuery({
    queryKey: ["workflow-revision", revisionId, "source-fit"],
    queryFn: ({ signal }) => api.workflowRevisionSourceFit(revisionId!, signal),
    enabled: Boolean(!previewContext && revisionId && sourceId),
    retry: false,
  });
  const intent = value?.request;
  // A preview within a submission asks the server about that submission, so
  // both ways are offered and the server answers for the one chosen.
  const modes: SourceFitMode[] = previewContext
    ? mode === "image" || mode === "auto" ? ["extend", "crop"] : []
    : revisionId && sourceId && capability.data?.available && !capability.isError
      ? capability.data.modes.filter((one) => one === "extend" || one === "crop") : [];
  const available = modes.length > 0;
  const valid = Boolean(intent && modes.includes(intent.mode) && sourceCanvasDimension(intent.width) && sourceCanvasDimension(intent.height));
  const context: SourceFitPreviewContext | undefined = previewContext
    ? JSON.parse(JSON.stringify({ ...previewContext, request: { ...previewContext.request, source_fit: intent } }))
    : undefined;
  const key = JSON.stringify([context, revisionId, sourceId, intent?.mode, intent?.width, intent?.height]);
  const previewQuery = useQuery({
    queryKey: ["source-canvas-preview", key],
    queryFn: ({ signal }) => context?.kind === "turn"
      ? api.previewTurnSourceFit(context.id, context.request, signal)
      : context?.kind === "prior-edit"
        ? api.previewPriorTurnSourceFit(context.id, context.request, signal)
        : api.previewWorkflowRevisionSourceFit(revisionId!, sourceId!, { ...intent! }, signal),
    enabled: available && valid && requestedKey === key,
    retry: false,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
  const answer = previewQuery.data;
  const expectedSource = context ? context.request.input_artifact_ids?.[0] : sourceId;
  const expectedRevision = context ? context.request.workflow_revision_id : revisionId;
  const matches = requestedKey === key && available && valid && !previewQuery.isFetching && !previewQuery.isError
    && (context || (value?.sourceArtifactId === sourceId && value?.workflowRevisionId === revisionId))
    && answer?.version === 1 && answer.mode === intent?.mode
    && Boolean(answer.workflow_revision_id && answer.source_artifact_id)
    && (!expectedRevision || answer.workflow_revision_id === expectedRevision)
    && (!expectedSource || answer.source_artifact_id === expectedSource)
    && answer.canvas?.width === intent?.width && answer.canvas?.height === intent?.height;
  const preview = matches ? answer : null;
  const selection: SourceFitSelection | null = preview && intent
    ? { sourceArtifactId: preview.source_artifact_id, workflowRevisionId: preview.workflow_revision_id, request: { ...intent } } : null;
  const choose = (next: SourceFitIntent | null) => {
    if (next === null) onChange(null);
    else if (context) onChange({ sourceArtifactId: sourceId ?? "", workflowRevisionId: revisionId ?? "", request: next });
    else if (sourceId && revisionId) onChange({ sourceArtifactId: sourceId, workflowRevisionId: revisionId, request: next });
  };
  const pending = requestedKey === key && previewQuery.isFetching;
  const requestPreview = () => {
    if (!available || !valid || !intent || pending) return;
    choose({ ...intent });
    if (requestedKey === key) void previewQuery.refetch();
    else setRequestedKey(key);
  };
  return {
    value, modes, available, preview, selection, choose, requestPreview, pending,
    invalidatePreview: () => setRequestedKey(null),
    canPreview: available && valid && !pending,
    checking: Boolean(!context && revisionId && sourceId && capability.isFetching),
    capabilityError: !context && capability.isError,
    missingRevision: !context && !revisionId,
    missingSource: !context && !sourceId,
    error: requestedKey === key && !previewQuery.isFetching
      && (previewQuery.isError || Boolean(answer && !matches)),
  };
}
