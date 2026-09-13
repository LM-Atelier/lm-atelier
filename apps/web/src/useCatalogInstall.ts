import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import type { CatalogModel, CatalogPreflight } from "./types";

export interface PendingInstall {
  model: CatalogModel;
  preflight: CatalogPreflight;
  installRole: string;
  engine: string;
  auxiliaryKind: "lora" | null;
}

/** Install a catalog item in two steps: check what it will cost, then confirm the transfer.
 *
 * Preflight and transfer are separate so a person sees the download size and
 * the checks before anything starts. `prepare` refuses with the blocking
 * reasons when the item cannot be installed safely; `confirm` starts the
 * download and refreshes the job list.
 */
export function useCatalogInstall() {
  const client = useQueryClient();
  const [pendingInstall, setPendingInstall] = useState<PendingInstall | null>(null);
  // Named as Models reports them in its FirstFailure list; the suggestions
  // panel shows both errors itself.
  const download = useMutation({
    mutationFn: async ({ model, selectedRole }: { model: CatalogModel; selectedRole: string }) => {
      const auxiliaryKind = selectedRole === "lora" ? "lora" : null;
      const installRole = auxiliaryKind ? "image" : selectedRole;
      const engine = model.required_runtime ?? (installRole === "chat" ? "llama.cpp" : "comfyui");
      // A CivitAI card's remote id is its exact version; that is also the
      // revision it pins. Hugging Face keeps floating "main".
      const revision = model.provider === "civitai" ? model.remote_id : "main";
      const preflight = auxiliaryKind
        ? await api.catalogPreflight(
            model.remote_id,
            installRole,
            engine,
            revision,
            [],
            auxiliaryKind,
            null,
            model.provider,
          )
        : await api.catalogPreflight(
            model.remote_id,
            installRole,
            engine,
            revision,
            [],
            null,
            // Preflight the exact workflow this card represents; a repository
            // can ship several and ranking must not answer for the user.
            model.workflow_template_id ?? null,
            model.provider,
          );
      if (!preflight.can_install) {
        const blockers = preflight.checks
          .filter((check) => check.status === "block")
          .map((check) => check.detail);
        throw new Error(blockers.join(" ") || "This model cannot be installed safely.");
      }
      if (!preflight.install_plan || preflight.install_plan.compatibility !== "supported") {
        throw new Error(
          preflight.install_plan?.failure_reason
          || "LM Atelier cannot safely activate this model with the current runtime.",
        );
      }
      return { model, preflight, installRole, engine, auxiliaryKind } satisfies PendingInstall;
    },
    onSuccess: (ready) => setPendingInstall(ready),
  });
  const confirmInstall = useMutation({
    mutationFn: ({ preflight, installRole, engine, auxiliaryKind }: PendingInstall) => {
      const downloadArguments = [
        preflight.remote_id,
        preflight.source_remote_id,
        installRole,
        engine,
        preflight.revision,
        preflight.selected_files,
        preflight.expected_sha256,
        preflight.file_sources ?? {},
        preflight.comfy_paths,
        preflight.workflow_template_id,
        preflight.workflow_template_sha256,
        preflight.install_plan?.id ?? null,
      ] as const;
      const contentRating = preflight.content_rating ?? "unknown";
      return auxiliaryKind
        ? api.download(...downloadArguments, auxiliaryKind, contentRating)
        : api.download(...downloadArguments, null, contentRating);
    },
    onSuccess: () => {
      setPendingInstall(null);
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  return {
    pendingInstall,
    cancel: () => setPendingInstall(null),
    prepare: download,
    confirm: confirmInstall,
  };
}
