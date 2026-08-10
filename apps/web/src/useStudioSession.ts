import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "./api";
import { generationIdentityFromProvenance } from "./generationIdentity";
import type { ChatDetail, GenerationIdentity, Message } from "./types";

const STUDIO_SESSION_KEY = "local-lm-studio-session";

/** The studio's server state: one hidden session and its result chain.
 *
 * The studio never renders a transcript, so this hook deliberately exposes
 * the session as a filmstrip - the ordered generated images with the source
 * first - rather than as messages. Applies are ordinary turns underneath;
 * the surface only ever sees pictures and the instruction that made each.
 */
export type StudioMaskUpload = {
  /** The mask raster as PNG alpha, at the source image's resolution. */
  blob: Blob;
  featherPx: number;
  invert: boolean;
};

type StudioApply = {
  instruction: string;
  artifactId: string;
  mask?: StudioMaskUpload;
  /** What the active tool asks for beyond words - a scale factor, say. */
  settings?: Record<string, unknown>;
  /** The workflow a recipe recorded, so applying one reproduces its run
   * rather than running its words against whatever is current. */
  workflowRevisionId?: string;
};

export type StudioStep = {
  messageId: string;
  artifactId: string;
  /** The instruction that produced this result; empty for the source. */
  instruction: string;
  isSource: boolean;
  generationIdentity: GenerationIdentity | null;
};

/** A session is only ever usable for the picture it was opened for.
 *
 * Storing the id alone let a switch between images pair the new source with
 * the previous session for as long as opening took: the filmstrip showed the
 * old chain under the new picture, and an Apply in that window was addressed
 * to the wrong session. Carrying the source alongside makes the mismatch
 * representable, so it can be refused instead of raced.
 */
type StudioSessionBinding = { id: string; source: string };

function storedBinding(): StudioSessionBinding | null {
  const raw = localStorage.getItem(STUDIO_SESSION_KEY);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (
      typeof parsed === "object" && parsed !== null
      && typeof (parsed as StudioSessionBinding).id === "string"
      && typeof (parsed as StudioSessionBinding).source === "string"
    ) {
      return parsed as StudioSessionBinding;
    }
  } catch {
    // A binding written before it carried its source. It cannot be matched
    // against anything, so it is not resumed; opening the image again costs
    // one request and cannot address the wrong session.
  }
  return null;
}

export function useStudioSession(sourceArtifactId: string | null, sourceChatId: string | null) {
  const client = useQueryClient();
  const [binding, setBinding] = useState<StudioSessionBinding | null>(storedBinding);
  // The only binding this render may act on. Anything else belongs to a
  // picture that is no longer open.
  const current = binding && binding.source === sourceArtifactId ? binding : null;
  const sessionId = current?.id ?? null;

  const open = useMutation({
    mutationFn: () => api.openStudioSession(sourceArtifactId!, sourceChatId),
    onSuccess: (session) => {
      const opened = { id: session.id, source: sourceArtifactId! };
      setBinding(opened);
      localStorage.setItem(STUDIO_SESSION_KEY, JSON.stringify(opened));
      client.setQueryData(["studio-session", session.id], session);
    },
  });

  // Opening an image is the studio's entry: find-or-create runs once per
  // source, and reopening the same image resumes its history.
  useEffect(() => {
    if (sourceArtifactId && !open.isPending) open.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceArtifactId, sourceChatId]);

  const session = useQuery({
    queryKey: ["studio-session", sessionId],
    queryFn: () => api.studioSession(sessionId!),
    enabled: Boolean(sessionId),
    refetchInterval: (query) => (hasPendingWork(query.state.data) ? 2_000 : false),
  });

  const apply = useMutation({
    mutationFn: async ({
      instruction,
      artifactId,
      mask,
      settings,
      workflowRevisionId,
    }: StudioApply) => {
      // Refused rather than raced. Between switching pictures and the new
      // session opening there is no session for what is on screen, and the
      // previous one is emphatically not it.
      if (!sessionId) {
        throw new Error("This picture is still opening. Try that again in a moment.");
      }
      // The mask uploads as its own artifact and travels in settings, not in
      // input_artifact_ids: a selection is instruction, not content, so it
      // must never render as an attachment or count toward edit lineage.
      const turnSettings = {
        ...settings,
        ...(mask ? { mask: await uploadMask(mask) } : {}),
      };
      return api.sendTurn(
        sessionId,
        instruction,
        "image",
        [artifactId],
        turnSettings,
        undefined,
        undefined,
        workflowRevisionId,
      );
    },
    onSuccess: () => void client.invalidateQueries({ queryKey: ["studio-session", sessionId] }),
  });

  return {
    sessionId,
    session: session.data ?? null,
    steps: session.data ? studioSteps(session.data, sourceArtifactId) : [],
    previewArtifactId: session.data ? studioPreviewArtifactId(session.data) : null,
    busy: open.isPending || apply.isPending || hasPendingWork(session.data),
    error: open.error ?? session.error ?? apply.error,
    /** `onAccepted` runs only once the turn has been taken.
     *
     * The surface clears the instruction and the selection there rather than
     * at dispatch: a refusal used to erase exactly the words that would have
     * been retried, leaving the reader to remember what they had typed.
     */
    apply: (
      instruction: string,
      artifactId: string,
      mask?: StudioMaskUpload,
      settings?: Record<string, unknown>,
      workflowRevisionId?: string,
      onAccepted?: () => void,
    ) =>
      apply.mutate(
        { instruction, artifactId, mask, settings, workflowRevisionId },
        { onSuccess: onAccepted },
      ),
  };
}

export const uploadMaskForTest = uploadMask;

async function uploadMask(mask: StudioMaskUpload) {
  const file = new File([mask.blob], "studio-selection.png", { type: "image/png" });
  const artifact = await api.upload(file);
  return {
    artifact_id: artifact.id,
    feather_px: mask.featherPx,
    invert: mask.invert,
  };
}

function hasPendingWork(session?: ChatDetail | null): boolean {
  return Boolean(session?.messages.some((message) => message.status === "pending"));
}

/** The newest provisional image from work that is still running.
 *
 * A preview is deliberately not a StudioStep: it is temporary server state,
 * not an edit result that can enter history or become the input to another
 * edit. */
export function studioPreviewArtifactId(session: ChatDetail): string | null {
  for (let index = session.messages.length - 1; index >= 0; index -= 1) {
    const message = session.messages[index];
    if (message.role !== "assistant" || message.status !== "pending") continue;
    const preview = message.parts.find(
      (part) =>
        part.type === "image"
        && Boolean(part.artifact_id)
        && part.metadata_json.preview === true,
    );
    if (preview?.artifact_id) return preview.artifact_id;
  }
  return null;
}

/** The filmstrip: the source, then one entry per generated result. */
export function studioSteps(
  session: ChatDetail,
  sourceArtifactId: string | null,
): StudioStep[] {
  const steps: StudioStep[] = [];
  if (sourceArtifactId) {
    steps.push({
      messageId: "source",
      artifactId: sourceArtifactId,
      instruction: "",
      isSource: true,
      generationIdentity: null,
    });
  }
  for (const message of session.messages) {
    if (message.role !== "assistant") continue;
    const image = message.parts.find(
      (part) => part.type === "image" && part.artifact_id && !part.metadata_json.preview,
    );
    if (!image?.artifact_id) continue;
    const metadata = message.parts.find((part) => part.type === "generation_metadata");
    steps.push({
      messageId: message.id,
      artifactId: image.artifact_id,
      instruction: instructionFor(session.messages, message),
      isSource: false,
      generationIdentity: generationIdentityFromProvenance(metadata?.metadata_json.provenance),
    });
  }
  return steps;
}

function instructionFor(messages: Message[], result: Message): string {
  const index = messages.findIndex((message) => message.id === result.id);
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    const message = messages[cursor];
    if (message.role !== "user") continue;
    return message.parts.find((part) => part.type === "text" && part.text)?.text ?? "";
  }
  return "";
}
