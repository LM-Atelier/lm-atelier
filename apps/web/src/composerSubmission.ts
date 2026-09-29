import { promptSourceForTurn, type ComposerDraft } from "./composerPromptSource";
import { mediaOutputCountForTurn } from "./mediaOutputCount";
import { survivingMentions, turnReferences, type TrackedMention } from "./mentionDraft";
import { normalizeSettingsForFields, resolveCapabilitySettings, resolveWorkflowSettings } from "./settings";
import type { EngineCapabilities, RoutingMode } from "./types";
import { roleForMode } from "./viewHelpers";

/** What a composer send carries in one mode, worked out in one place.

The send and the source canvas preview both read it, so a canvas is previewed
against the same text, inputs and settings the send then carries. */
export function composerSubmission({
  mode, engines, workflowSchema, acceptsAddedLoras, settings, templateSettings, text, mentions, draft, inputCount,
  outputCount, accepting,
}: {
  mode: RoutingMode;
  engines: EngineCapabilities[];
  workflowSchema: Record<string, unknown> | undefined;
  acceptsAddedLoras: boolean;
  settings: Record<string, unknown>;
  templateSettings: { settings: Record<string, unknown> } | null | undefined;
  text: string;
  mentions: TrackedMention[];
  draft: ComposerDraft;
  inputCount: number;
  outputCount: number;
  /** An accepted edit keeps its chosen settings whole; a send keeps only fields its mode offers. */
  accepting: boolean;
}) {
  const role = roleForMode(mode);
  const fields = resolveWorkflowSettings(
    resolveCapabilitySettings(engines.find((item) => item.roles.includes(role)), role), workflowSchema, acceptsAddedLoras,
  );
  const chosenSettings = { ...settings, ...templateSettings?.settings };
  const requestedOutputCount = mediaOutputCountForTurn(mode, outputCount);
  const references = turnReferences(survivingMentions(text, mentions));
  const promptSource = promptSourceForTurn(draft, mode, inputCount, references.length, requestedOutputCount);
  const sentSettings = accepting ? chosenSettings : mode === "auto" ? {} : normalizeSettingsForFields(chosenSettings, fields);
  return { fields, chosenSettings, requestedOutputCount, references, promptSource, settings: sentSettings };
}
