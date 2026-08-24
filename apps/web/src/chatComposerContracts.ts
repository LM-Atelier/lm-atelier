import type { VisualTarget } from "./libraryEditTargets";
import type { TurnReference } from "./mentionDraft";
import type {
  ComposerDraft,
  ComposerDraftUpdate,
  ComposerPromptSource,
} from "./composerPromptSource";
import type {
  ChatDetail,
  EngineCapabilities,
  EngineRole,
  GenerationPreset,
  ModelProfile,
  Project,
  RoutingMode,
  Workflow,
  WorkPlan,
} from "./types";

type SendFromComposer = (
  text: string,
  mode: RoutingMode,
  artifacts: string[],
  settings: Record<string, unknown>,
  references: TurnReference[],
  outputCount?: number,
  promptSource?: ComposerPromptSource,
) => void;

export type PendingTurn = { id: string; text: string; mode: RoutingMode };

export interface ComposerProps {
  chat: ChatDetail;
  engines: EngineCapabilities[];
  profiles: ModelProfile[];
  stoppable: boolean;
  settings: Record<string, unknown>;
  onSettings: (settings: Record<string, unknown>) => void;
  settingsRole: EngineRole;
  onSettingsRole: (role: EngineRole) => void;
  presets: GenerationPreset[];
  presetId: string | null;
  onPreset: (presetId: string | null) => void;
  onMode: (mode: RoutingMode) => void;
  onSend: SendFromComposer;
  onStop: () => void;
  onStopAndSend: SendFromComposer;
  maxMediaOutputsPerPlan: number;
  workflows: Workflow[];
  project?: Project;
  visualTarget?: VisualTarget | null;
  quoteTarget?: { text: string; requestId: number } | null;
  draft: ComposerDraft;
  onDraftChange: (update: ComposerDraftUpdate) => void;
}

export interface ChatViewProps {
  onOpenStudio: (artifactId: string) => void;
  chat?: ChatDetail;
  engines: EngineCapabilities[];
  profiles: ModelProfile[];
  workflows: Workflow[];
  project?: Project;
  liveText: Record<string, string>;
  pendingTurns: PendingTurn[];
  workPlans: WorkPlan[];
  settings: Record<string, unknown>;
  settingsRole: EngineRole;
  onSettingsRole: (role: EngineRole) => void;
  presets: GenerationPreset[];
  presetId: string | null;
  onSettings: (settings: Record<string, unknown>) => void;
  onPreset: (presetId: string | null) => void;
  onMode: (mode: RoutingMode) => void;
  onSend: SendFromComposer;
  onRegenerate: (messageId: string, settings: Record<string, unknown>) => void;
  onSelectRevision: (messageId: string, revisionId: string) => void;
  onEdit: (
    messageId: string,
    text: string,
    mode: RoutingMode,
    settings: Record<string, unknown>,
  ) => void;
  onStop: () => void;
  onStopAndSend: SendFromComposer;
  maxMediaOutputsPerPlan: number;
  onCancelPlan: (planId: string) => void;
  onCancelStep: (stepId: string) => void;
  onRetryStep: (stepId: string) => void;
  onDeleteExchange: (messageId: string) => void;
  onForkThread: (messageId: string) => void;
  libraryEdit?: VisualTarget | null;
  composerDraft: ComposerDraft;
  onComposerDraft: (update: ComposerDraftUpdate) => void;
}
