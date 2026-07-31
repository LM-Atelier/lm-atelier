import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { RoutingMode } from "./types";

/** How long the draft must stop changing before it is classified. */
export const DRAFT_SETTLE_MS = 200;

/**
 * Whether the draft in the composer would reuse the most recent visual.
 *
 * This is the router's decision, so it is asked for rather than reproduced. The
 * browser used to keep its own copy of the router's patterns, which drifted
 * every time the server learned a new phrasing: by the time it was replaced it
 * misread 13 of the 17 wordings the server routes to an image edit, so the
 * composer showed the wrong workflow schema and hid the edit-strength control
 * while the server performed an edit anyway.
 *
 * Nothing is requested until a prior visual actually exists and the draft is
 * non-empty, so an ordinary text conversation never asks.
 */
export function useDraftClassification(
  chatId: string,
  text: string,
  mode: RoutingMode,
  hasPriorVisual: boolean,
): boolean {
  const [settled, setSettled] = useState(text);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(text), DRAFT_SETTLE_MS);
    return () => window.clearTimeout(timer);
  }, [text]);

  const enabled = hasPriorVisual && settled.trim().length > 0;
  const classification = useQuery({
    queryKey: ["classify-draft", chatId, mode, settled],
    queryFn: () => api.classifyDraft(chatId, settled, mode),
    enabled,
    staleTime: 5 * 60 * 1000,
    // Hold the previous answer while the next is in flight, so the
    // edit-strength control does not flicker as the user types.
    placeholderData: (previous) => previous,
  });

  return enabled && (classification.data?.references_prior_visual ?? false);
}
