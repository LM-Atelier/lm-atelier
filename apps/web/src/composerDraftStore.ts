/** How the composer's draft and the workspace's stored draft map onto each other. */

import type { ComposerDraft } from "./composerPromptSource";
import type { Artifact, ChatComposerDraft, ChatComposerDraftInput } from "./types";
import { initialTurnEditorState } from "./useTurnEditorState";

/** How long typing must pause before the draft is saved. */
export const DRAFT_SAVE_DELAY_MS = 800;

/** What the workspace stores for a draft, from what the composer holds. */
export function storedDraft(draft: ComposerDraft): ChatComposerDraftInput {
  const editor = draft.editor;
  return {
    text: draft.text,
    prompt_source: draft.promptSource,
    mode: editor?.mode ?? "auto",
    output_count: editor?.outputCount ?? 1,
    attachments: (editor?.attachments ?? []).map((attachment) => ({
      artifact_id: attachment.id,
      kind: attachment.kind,
      origin: attachment.origin,
    })),
    mentions: (editor?.mentions ?? []).map((mention) => ({
      reference_subject_id: mention.referenceSubjectId,
      mention_slug: mention.mentionSlug,
    })),
    template_settings: editor?.templateSettings ?? null,
  };
}

/** Whether a draft holds anything worth keeping. */
export function draftHasContent(draft: ChatComposerDraftInput): boolean {
  return Boolean(
    draft.text.trim()
      || draft.prompt_source
      || draft.attachments.length
      || draft.mentions.length
      || draft.template_settings,
  );
}

/** The composer's draft, from what the workspace stored.
 *
 * `artifacts` supplies each attachment's stored file record where it was read,
 * so a restored attachment shows its name rather than its identifier.
 */
export function composerDraftFrom(
  stored: ChatComposerDraft,
  artifacts: ReadonlyMap<string, Artifact> = new Map(),
): ComposerDraft {
  return {
    text: stored.text,
    promptSource: stored.prompt_source,
    editor: {
      ...initialTurnEditorState(stored.mode, undefined),
      outputCount: stored.output_count,
      attachments: stored.attachments.map((attachment) => ({
        id: attachment.artifact_id,
        kind: attachment.kind,
        artifact: artifacts.get(attachment.artifact_id) ?? null,
        origin: attachment.origin,
      })),
      mentions: stored.mentions.map((mention) => ({
        referenceSubjectId: mention.reference_subject_id,
        mentionSlug: mention.mention_slug,
      })),
      templateSettings: stored.template_settings,
    },
  };
}
