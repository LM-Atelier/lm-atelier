import { useCallback, type ReactNode } from "react";
import type { TurnConfirmationHandler, TurnConfirmationRequest } from "./api";
import { formatBytes } from "./format";
import { useConfirm } from "./useConfirm";

function detailFor(request: TurnConfirmationRequest) {
  if (request.kind === "ordered_plan") {
    return (
      <ul className="confirm-details">
        <li>Sequence: {request.details.sequence.join(" \u2192 ")}</li>
        {request.details.videoDurationSeconds !== undefined && (
          <li>Video duration: about {request.details.videoDurationSeconds} seconds</li>
        )}
        {request.details.estimatedWorkingBytes !== undefined && (
          <li>Working space: up to {formatBytes(request.details.estimatedWorkingBytes)}</li>
        )}
      </ul>
    );
  }
  return (
    <ul className="confirm-details">
      <li>Suggested operation: {request.details.operation}</li>
      {request.details.durationSeconds !== undefined && (
        <li>Output duration: about {request.details.durationSeconds} seconds</li>
      )}
      {request.details.estimatedIntermediateBytes !== undefined && (
        <li>Intermediate space: up to {formatBytes(request.details.estimatedIntermediateBytes)}</li>
      )}
    </ul>
  );
}

export function useTurnConfirmation(): [ReactNode, TurnConfirmationHandler] {
  const [dialog, confirm] = useConfirm();
  const requestConfirmation = useCallback((request: TurnConfirmationRequest) => confirm({
    title: request.title,
    question: request.question,
    confirmLabel: request.confirmLabel,
    tone: "action",
    detail: detailFor(request),
  }), [confirm]);
  return [dialog, requestConfirmation];
}
