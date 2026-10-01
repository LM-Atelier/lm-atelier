import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { EnlargementPreview } from "./studioEnlargement";
import { buildTurnRequest } from "./turnRequest";

/** Where an Enhance stands before Apply: which workflow it runs, or why it cannot yet. */
export type StudioEnlargement = {
  preview: EnlargementPreview | null;
  checking: boolean;
  /** The server's reason, said as it said it, when no workflow can take the enlargement. */
  error: string | null;
  refresh: () => void;
};

/** The query root of every enlargement preview, asked again whenever the answer may have changed. */
export const ENLARGEMENT_PREVIEW_ROOT = "studio-enlargement";

/** Reads whose change can change which workflow enlarges, or the default it enlarges by.
 *
 * A scoped recipe is one of them, and choosing or editing one names no
 * workflow revision the preview's own key could follow, so a change to either
 * read asks the preview again.
 */
const RECIPE_ROOTS: ReadonlySet<unknown> = new Set(["workflow-recipe-choice", "workflow-recipes"]);

/** Asking which workflow an Enhance of this picture runs, and what it lets the person choose.
 *
 * Asked only while Enhance is the tool in hand, and asked again whenever the
 * picture, the Studio's workflow choice, its recipe, or any scoped recipe
 * changes, since each can change the answer. The answer authorizes nothing:
 * Apply names the workflow it returns, and the server checks that again when
 * the edit is taken. While it is being asked again, or when asking again
 * fails, there is no answer to apply.
 */
export function useStudioEnlargement({
  sessionId,
  artifactId,
  active,
  words,
  recipeRevisionId,
}: {
  sessionId: string | null;
  artifactId: string | null;
  active: boolean;
  words: string;
  /** The workflow a chosen recipe recorded, which the enlargement then runs. */
  recipeRevisionId?: string;
}): StudioEnlargement {
  const client = useQueryClient();
  // The same read the workflow chooser makes, so a new choice there is a new question here.
  const selections = useQuery({
    queryKey: ["chat", sessionId, "workflow-selections"],
    queryFn: () => api.chatWorkflowSelections(sessionId!),
    enabled: active && Boolean(sessionId),
  });
  const choice = JSON.stringify(selections.data?.find((one) => one.selector_capability === "image") ?? null);
  const preview = useQuery({
    queryKey: [ENLARGEMENT_PREVIEW_ROOT, sessionId, artifactId, recipeRevisionId ?? null, choice],
    queryFn: ({ signal }) => api.previewEnlargement(sessionId!, buildTurnRequest({
      text: words,
      mode: "image",
      inputArtifactIds: [artifactId!],
      settings: {},
      workflowRevisionId: recipeRevisionId,
      upscale: true,
    }), signal),
    enabled: active && Boolean(sessionId && artifactId) && !selections.isLoading,
    retry: false,
  });
  useEffect(() => {
    if (!active) return;
    return client.getQueryCache().subscribe((event) => {
      // A recipe read that has new data, or is marked out of date, may mean a new answer.
      if (
        event.type === "updated"
        && (event.action.type === "success" || event.action.type === "invalidate")
        && RECIPE_ROOTS.has(event.query.queryKey[0])
      ) {
        void client.invalidateQueries({ queryKey: [ENLARGEMENT_PREVIEW_ROOT] });
      }
    });
  }, [active, client]);
  const answer = active && !preview.isError && !preview.isFetching
    && preview.data?.version === 1 && preview.data.status === "ready"
    ? preview.data
    : null;
  return {
    preview: answer,
    checking: active && !preview.isError && (selections.isLoading || preview.isFetching || preview.isPending),
    error: active && preview.isError ? (preview.error as Error).message : null,
    // A refetch runs even while the query is switched off, so a refused edit
    // made with another tool must not ask about an enlargement.
    refresh: () => {
      if (active) void preview.refetch();
    },
  };
}
