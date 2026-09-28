import { useState } from "react";
import { ErrorCallout } from "./ErrorCallout";
import type { SettingField } from "./types";
import { useWorkflowLoraControls } from "./useWorkflowLoraControls";

/** Keep a selected stack until the workflow can say whether it accepts added LoRAs. */
export function useComposerLoraControls(revisionId: string | null) {
  const { controls, unavailable, retry } = useWorkflowLoraControls(revisionId);
  const [blocked, setBlocked] = useState(false);
  return {
    acceptsAddedLoras: controls?.accepts_added_loras ?? false,
    canSend(values: Record<string, unknown>, fields: SettingField[]): boolean {
      const waiting = Boolean(revisionId && !controls
        && Array.isArray(values.loras) && values.loras.length > 0
        && !fields.some((field) => field.key === "loras"));
      setBlocked(waiting);
      return !waiting;
    },
    error: blocked && revisionId && !controls ? <ErrorCallout
      message={unavailable ? "The selected workflow's LoRA controls could not be loaded. Your choices are kept." : "Wait for the selected workflow's LoRA controls to load. Your choices are kept."}
      action={unavailable ? <button type="button" onClick={retry}>Retry LoRA controls</button> : undefined}
    /> : null,
  };
}
