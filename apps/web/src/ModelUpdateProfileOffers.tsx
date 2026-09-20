import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import type { ModelUpdateDownload } from "./modelUpdateDownloads";
import type { ModelProfile } from "./types";

export function ModelUpdateProfileOffers({ downloads, onDismiss }: {
  downloads: ModelUpdateDownload[];
  onDismiss: (jobId: string) => void;
}) {
  return downloads.map((download) => <UpdateOffer key={download.jobId} download={download} onDismiss={onDismiss} />);
}

function UpdateOffer({ download, onDismiss }: {
  download: ModelUpdateDownload;
  onDismiss: (jobId: string) => void;
}) {
  const client = useQueryClient();
  const [switched, setSwitched] = useState<string[]>([]);
  const job = useQuery({
    queryKey: ["model-update-download", download.jobId],
    queryFn: () => api.downloadJob(download.jobId),
    retry: false,
    refetchInterval: (query) => ["complete", "failed", "cancelled"].includes(query.state.data?.status ?? "") ? false : 3_000,
  });
  const complete = job.data?.status === "complete";
  const installId = complete && typeof job.data?.result_json.model_install_id === "string"
    ? job.data.result_json.model_install_id : null;
  const model = useQuery({
    queryKey: ["models", installId],
    queryFn: () => api.modelInstall(installId!),
    enabled: installId !== null,
    retry: false,
  });
  const ready = model.data?.active && model.data.readiness === "ready";
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles, enabled: Boolean(ready) });
  const change = useMutation({
    mutationFn: (profile: ModelProfile) => api.updateProfileModel(profile.id, {
      expected_install_id: download.previousInstallId, download_job_id: download.jobId,
    }),
    onSuccess: (updated) => {
      setSwitched((current) => [...current, updated.name]);
      client.setQueryData<ModelProfile[]>(["profiles"], (current) => current?.map((profile) => profile.id === updated.id ? updated : profile));
      for (const key of ["profiles", "models", "workflow-families", "setup-readiness"]) {
        void client.invalidateQueries({ queryKey: [key] });
      }
    },
  });
  const terminal = complete || ["failed", "cancelled"].includes(job.data?.status ?? "");
  if (!terminal && !job.error) return null;
  const eligible = (profiles.data ?? []).filter((profile) => (
    profile.model_install_id === download.previousInstallId
    && profile.role === model.data?.role && profile.engine === model.data.engine
  ));
  const error = job.error ?? model.error ?? profiles.error ?? change.error;
  return <section className="model-updates" aria-label={`Update ${download.modelName}`}>
    <h2>{download.modelName}</h2>
    {!complete && !job.error && <p>The update did not finish installing. Your profiles have not been switched.</p>}
    {complete && !model.isLoading && !ready && !model.error && <p>The update needs current runtime verification before a profile can switch to it.</p>}
    {ready && <>
      <p>The update passed its runtime checks. Switch a profile to use it for future work; the previous version stays installed.</p>
      {eligible.length === 0 && !profiles.isLoading && switched.length === 0 && <p>No profiles use the previous version.</p>}
    </>}
    {switched.length > 0 && <p role="status">Updated {switched.join(", ")}.</p>}
    <ErrorCallout message={error instanceof Error ? error.message : null} />
    <div className="storage-actions">{ready && eligible.map((profile) => <button
        key={profile.id} type="button" className="primary compact-button"
        aria-disabled={change.isPending}
        onClick={() => { if (!change.isPending) change.mutate(profile); }}
      >Switch {profile.name}</button>)}
    <button type="button" className="secondary compact-button" aria-disabled={change.isPending}
      onClick={() => { if (!change.isPending) onDismiss(download.jobId); }}>Dismiss update</button>
    </div>
  </section>;
}
