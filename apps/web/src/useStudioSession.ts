import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "./api";
import type { ChatDetail, Message } from "./types";

const STUDIO_SESSION_KEY = "local-lm-studio-session";

/** The studio's server state: one hidden session and its result chain.
 *
 * The studio never renders a transcript, so this hook deliberately exposes
 * the session as a filmstrip - the ordered generated images with the source
 * first - rather than as messages. Applies are ordinary turns underneath;
 * the surface only ever sees pictures and the instruction that made each.
 */
export type StudioStep = {
  messageId: string;
  artifactId: string;
  /** The instruction that produced this result; empty for the source. */
  instruction: string;
  isSource: boolean;
};

export function useStudioSession(sourceArtifactId: string | null, sourceChatId: string | null) {
  const client = useQueryClient();
  const [sessionId, setSessionId] = useState<string | null>(
    () => localStorage.getItem(STUDIO_SESSION_KEY),
  );

  const open = useMutation({
    mutationFn: () => api.openStudioSession(sourceArtifactId!, sourceChatId),
    onSuccess: (session) => {
      setSessionId(session.id);
      localStorage.setItem(STUDIO_SESSION_KEY, session.id);
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
    mutationFn: ({ instruction, artifactId }: { instruction: string; artifactId: string }) =>
      api.sendTurn(sessionId!, instruction, "image", [artifactId], {}),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["studio-session", sessionId] }),
  });

  return {
    sessionId,
    session: session.data ?? null,
    steps: session.data ? studioSteps(session.data, sourceArtifactId) : [],
    busy: apply.isPending || hasPendingWork(session.data),
    error: open.error ?? session.error ?? apply.error,
    apply: (instruction: string, artifactId: string) => apply.mutate({ instruction, artifactId }),
  };
}

function hasPendingWork(session?: ChatDetail | null): boolean {
  return Boolean(session?.messages.some((message) => message.status === "pending"));
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
    });
  }
  for (const message of session.messages) {
    if (message.role !== "assistant") continue;
    const image = message.parts.find(
      (part) => part.type === "image" && part.artifact_id && !part.metadata_json.preview,
    );
    if (!image?.artifact_id) continue;
    steps.push({
      messageId: message.id,
      artifactId: image.artifact_id,
      instruction: instructionFor(session.messages, message),
      isSource: false,
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
