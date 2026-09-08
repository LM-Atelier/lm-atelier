import { useCallback, useEffect, useRef, useState, type SetStateAction } from "react";
import type { RoutingMode } from "./types";
import type { ComposerAttachment } from "./useComposerUploads";
import type { TrackedMention } from "./mentionDraft";

/** State that belongs to one draft, including an unacknowledged submission. */
export interface TurnEditorState {
  requestId: string;
  mode: RoutingMode;
  attachments: ComposerAttachment[];
  attachmentIntent: "inherit" | "replace";
  mentions: TrackedMention[];
  referenceIntent: "inherit" | "replace";
  outputCount: number;
  templateSettings: { name: string; settings: Record<string, unknown> } | null;
  submittedFingerprint?: string;
}

export function useTurnEditorState(
  initialMode: RoutingMode,
  onMode: (mode: RoutingMode) => void,
  initialState: Partial<TurnEditorState> | undefined,
  controlledState: TurnEditorState | undefined,
  onChange: ((update: SetStateAction<TurnEditorState>) => void) | undefined,
) {
  const [localState, setLocalState] = useState<TurnEditorState>(() => ({
    requestId: crypto.randomUUID(),
    mode: initialMode,
    attachments: [],
    attachmentIntent: "replace",
    mentions: [],
    referenceIntent: "replace",
    outputCount: 1,
    templateSettings: null,
    ...initialState,
  }));
  const state = controlledState ?? localState;
  const updateState = onChange ?? setLocalState;
  const selectedMode = useRef(state.mode);
  useEffect(() => { selectedMode.current = state.mode; }, [state.mode]);
  const currentMode = () => selectedMode.current;
  const setOutputCount = (value: number) => updateState((current) => ({ ...current, outputCount: value }));
  const changeMode = useCallback((value: RoutingMode) => {
    selectedMode.current = value;
    updateState((current) => ({ ...current, mode: value }));
    onMode(value);
  }, [onMode, updateState]);
  const setTemplateSettings = (value: TurnEditorState["templateSettings"]) => updateState((current) => ({ ...current, templateSettings: value }));
  const setAttachments = useCallback((update: SetStateAction<ComposerAttachment[]>) => updateState((current) => ({
    ...current,
    attachmentIntent: "replace",
    attachments: typeof update === "function" ? update(current.attachments) : update,
  })), [updateState]);
  return { state, updateState, setOutputCount, changeMode, currentMode, setTemplateSettings, setAttachments };
}
