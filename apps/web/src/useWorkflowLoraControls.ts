import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { WorkflowLoraControls } from "./types";

async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** The LoRAs one exact workflow revision applies, or nothing while that is not known.
 *
 * The answer names its revision only by digest, so it is checked against the
 * revision asked for before it is shown. Only a successful answer is returned:
 * when a refresh fails, the previous answer the query still holds is not lent,
 * because nothing then says it is still true.
 */
export function useWorkflowLoraControls(revisionId: string | null): {
  controls: WorkflowLoraControls | null;
  unavailable: boolean;
} {
  const query = useQuery({
    queryKey: ["workflows", "lora-controls", revisionId],
    queryFn: async ({ signal }) => {
      const controls = await api.workflowLoraControls(revisionId!, signal);
      if (controls.revision_scope_sha256 !== await sha256Hex(revisionId!)) {
        throw new Error("The workflow's LoRAs could not be read.");
      }
      return controls;
    },
    enabled: revisionId !== null,
  });
  return {
    controls: query.isSuccess ? query.data : null,
    unavailable: query.isError,
  };
}
