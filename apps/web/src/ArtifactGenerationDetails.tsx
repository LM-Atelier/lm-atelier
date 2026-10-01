import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { RecordedGenerationConfiguration } from "./GenerationDetails";

function ArtifactConfiguration({ artifactId }: { artifactId: string }) {
  const record = useQuery({
    queryKey: ["artifact-generation-details", artifactId],
    queryFn: async () => {
      const artifact = await api.artifact(artifactId);
      const runId = artifact.metadata_json.run_id;
      if (typeof runId !== "string" || !runId) return null;
      const run = await api.run(runId);
      const outputs = run.provenance_json.outputs;
      if (!Array.isArray(outputs) || !outputs.some((output) =>
        output !== null && typeof output === "object" && output.artifact_id === artifactId,
      )) return null;
      return run.provenance_json;
    },
    retry: false,
    staleTime: Infinity,
  });
  if (record.isPending) return <p role="status">Loading generation details…</p>;
  if (record.isError) return <p role="alert">Generation details could not be loaded.</p>;
  return <RecordedGenerationConfiguration provenance={record.data} />;
}

export function ArtifactGenerationDetails({ artifactId }: { artifactId: string }) {
  const [open, setOpen] = useState(false);
  return <details className="generation-details" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Generation details</summary>
    {open && <ArtifactConfiguration artifactId={artifactId} />}
  </details>;
}
