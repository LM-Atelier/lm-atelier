import type { SourceFitSelection } from "./sourceFit";
import type { TranscriptReads } from "./useChatTranscript";
import type { VisualTarget } from "./libraryEditTargets";
import type { TurnReference } from "./mentionDraft";
import type {
  ComposerDraft,
  ComposerDraftUpdate,
  ComposerPromptSource,
} from "./composerPromptSource";
import type {
  ChatDetail,
  ChatTranscriptContext,
  EngineCapabilities,
  EngineRole,
  ImageInputRole,
  Project,
  PriorTurnEditAccepted,
  RoutingMode,
  WorkPlan,
} from "./types";

export type SendFromComposer = (
  text: string,
  mode: RoutingMode,
  artifacts: string[],
  settings: Record<string, unknown>,
  references: TurnReference[],
  outputCount?: number,
  promptSource?: ComposerPromptSource,
  sourceFit?: SourceFitSelection,
  imageRoles?: ImageInputRole[],
) => void;

export type PendingTurn = { id: string; text: string; mode: RoutingMode };

export interface ComposerProps {
  chat: ChatDetail;
  transcriptContext?: ChatTranscriptContext;
  engines: EngineCapabilities[];
  stoppable: boolean;
  settings: Record<string, unknown>;
  onSettings: (settings: Record<string, unknown>, changedKeys?: string[]) => void;
  settingsRole: EngineRole;
  onSettingsRole: (role: EngineRole) => void;
  presetId: string | null;
  onPreset: (presetId: string | null) => void;
  onMode: (mode: RoutingMode) => void;
  onSend: SendFromComposer;
  onStop: () => void;
  onStopAndSend: SendFromComposer;
  maxMediaOutputsPerPlan: number;
  project?: Project;
  visualTarget?: VisualTarget | null;
  quoteTarget?: { text: string; requestId: number } | null;
  draft: ComposerDraft;
  onDraftChange: (update: ComposerDraftUpdate) => void;
}

export interface ChatViewProps {
  transcript?: TranscriptReads;
  onOpenStudio: (artifactId: string) => void;
  chat?: ChatDetail;
  engines: EngineCapabilities[];
  project?: Project;
  liveText: Record<string, string>;
  pendingTurns: PendingTurn[];
  workPlans: WorkPlan[];
  settings: Record<string, unknown>;
  settingsRole: EngineRole;
  onSettingsRole: (role: EngineRole) => void;
  presetId: string | null;
  onSettings: (settings: Record<string, unknown>, changedKeys?: string[]) => void;
  onPreset: (presetId: string | null) => void;
  onMode: (mode: RoutingMode) => void;
  onSend: SendFromComposer;
  onRegenerate: (messageId: string, settings: Record<string, unknown>) => void;
  onSelectRevision: (messageId: string, revisionId: string) => void;
  onEditAccepted: (accepted: PriorTurnEditAccepted) => void;
  onStop: () => void;
  onStopAndSend: SendFromComposer;
  maxMediaOutputsPerPlan: number;
  onCancelPlan: (planId: string) => void;
  onRetryPlan: (planId: string) => void;
  onCancelStep: (stepId: string) => void;
  onRetryStep: (stepId: string) => void;
  onDeleteExchange: (messageId: string) => void;
  onRemoveItem: (messageId: string) => void;
  onForkThread: (messageId: string) => void;
  libraryEdit?: VisualTarget | null;
  /** Called once the chat has taken `libraryEdit`, so whoever handed it down can let it go. */
  onLibraryEditTaken?: () => void;
  composerDraft: ComposerDraft;
  onComposerDraft: (update: ComposerDraftUpdate) => void;
}
