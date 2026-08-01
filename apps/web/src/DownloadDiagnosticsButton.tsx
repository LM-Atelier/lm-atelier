import { useMutation } from "@tanstack/react-query";

import { api } from "./api";

// The diagnostics bundle is redacted server-side (no prompts, media, tokens,
// or absolute paths) and now carries worker state, exit codes, and the same
// redacted failure tails the screen shows.
export function DownloadDiagnosticsButton() {
  const download = useMutation({
    mutationFn: () => api.createDiagnostics(),
    onSuccess: (artifact) => {
      const link = document.createElement("a");
      link.href = artifact.url;
      link.download = "";
      link.click();
    },
  });
  return (
    <button
      className="secondary"
      disabled={download.isPending}
      title="Saves a redacted report of application and worker state for a bug report"
      onClick={() => download.mutate()}
    >
      {download.isPending ? "Preparing…" : "Download diagnostics"}
    </button>
  );
}
