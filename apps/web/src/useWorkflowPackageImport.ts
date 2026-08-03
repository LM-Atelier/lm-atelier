import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "./api";
import type { WorkflowBundle, WorkflowPackageAnalysis } from "./types";

export type PackageReviewState = {
  analysis: WorkflowPackageAnalysis;
  fileName: string;
  /** The parsed export itself, so a ready review can import it directly. */
  uiGraph: Record<string, unknown>;
};

// FileReader rather than File.text(): equally supported in browsers, and it
// also exists in the test DOM, so this path is actually exercised by tests.
function readFileText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the file"));
    reader.readAsText(file);
  });
}

/** Import a workflow file, or review it when it is not importable yet.
 *
 * LM Atelier bundles import directly, as before. A raw ComfyUI export used
 * to be rejected with "not a bundle" - now it goes to the analyzer and comes
 * back as a review of what it would need, with nothing persisted, executed,
 * or trusted. Anything else is still an error.
 */
export function useWorkflowPackageImport(onImported: () => void) {
  const [packageReview, setPackageReview] = useState<PackageReviewState | null>(null);
  const importMutation = useMutation({
    mutationFn: async (file: File) => {
      const parsed = JSON.parse(await readFileText(file)) as Record<string, unknown>;
      if (parsed.format === "lm-atelier-workflow") {
        return api.importWorkflow(parsed as unknown as WorkflowBundle);
      }
      if (Array.isArray(parsed.nodes)) {
        setPackageReview({
          analysis: await api.analyzeWorkflowPackage(parsed),
          fileName: file.name,
          uiGraph: parsed,
        });
        return null;
      }
      throw new Error(
        "This is not an LM Atelier workflow bundle or a ComfyUI workflow export.",
      );
    },
    onSuccess: (imported) => {
      if (imported) onImported();
    },
  });
  return {
    importFile: (file?: File) => {
      if (file) importMutation.mutate(file);
    },
    importError: importMutation.error,
    packageReview,
    closePackageReview: () => setPackageReview(null),
  };
}
