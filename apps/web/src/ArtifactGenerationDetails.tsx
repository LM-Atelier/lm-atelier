import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { RecordedGenerationConfiguration } from "./GenerationDetails";
import { VideoUtilityOriginDetails } from "./VideoUtilityOriginDetails";
import { videoUtilityOrigin } from "./videoUtilityText";

function ArtifactConfiguration({ artifactId }: { artifactId: string }) {
  const record = useQuery({
    queryKey: ["artifact-generation-details", artifactId],
    queryFn: async () => {
      const artifact = await api.artifact(artifactId);
      const origin = videoUtilityOrigin(artifact.metadata_json);
      if (origin) {
        // A source can be removed once nothing keeps it, so its absence is an answer.
        const sourceName = await api.artifact(origin.sourceId).then(
          (source) => source.original_name ?? "",
          (error: unknown) => {
            if (error instanceof ApiError && error.status === 404) return null;
            throw error;
          },
        );
        return { kind: "utility", origin, sourceName } as const;
      }
      const runId = artifact.metadata_json.run_id;
      if (typeof runId !== "string" || !runId) return { kind: "run", provenance: null } as const;
      const run = await api.run(runId);
      const outputs = run.provenance_json.outputs;
      if (!Array.isArray(outputs) || !outputs.some((output) =>
        output !== null && typeof output === "object" && output.artifact_id === artifactId,
      )) return { kind: "run", provenance: null } as const;
      return { kind: "run", provenance: run.provenance_json } as const;
    },
    retry: false,
    staleTime: Infinity,
  });
  if (record.isPending) return <p role="status">Loading generation details…</p>;
  if (record.isError) return <p role="alert">Generation details could not be loaded.</p>;
  if (record.data.kind === "utility") {
    return <VideoUtilityOriginDetails origin={record.data.origin} sourceName={record.data.sourceName} />;
  }
  return <RecordedGenerationConfiguration provenance={record.data.provenance} />;
}

export function ArtifactGenerationDetails({ artifactId }: { artifactId: string }) {
  const [open, setOpen] = useState(false);
  return <details className="generation-details" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Generation details</summary>
    {open && <ArtifactConfiguration artifactId={artifactId} />}
  </details>;
}
