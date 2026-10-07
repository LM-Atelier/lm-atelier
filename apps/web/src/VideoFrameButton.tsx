import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { Camera } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { ApiError, api } from "./api";
import { ShieldedMedia } from "./ShieldedMedia";
import type { VideoUtilityLimit } from "./types";

/** Why a frame cannot be saved, as the person is told it. */
const LIMIT_TEXT: Record<VideoUtilityLimit, string> = {
  "video-codec-unsupported": "Its video is stored in a format the video utilities do not decode.",
  "audio-codec-unsupported": "Its sound cannot be copied unchanged.",
  "video-duration-unknown": "It does not say how long it is.",
  "video-duration-too-long": "It is longer than the video utilities read.",
  "video-frame-too-large": "Its frames are larger than the video utilities read.",
  "video-rotation-unsupported": "It is turned by an angle other than a quarter turn.",
};
const FINISHED = new Set(["complete", "failed", "cancelled", "interrupted"]);

function seconds(value: number): string {
  return value.toFixed(3) + " s";
}

function numberField(record: Record<string, unknown>, key: string): number | null {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Pick a moment in a stored video and save the frame there as a new picture. */
function VideoFrameBody({ artifactId, source }: { artifactId: string; source: string }) {
  const video = useRef<HTMLVideoElement>(null);
  const [at, setAt] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const probe = useQuery({
    queryKey: ["video-probe", artifactId],
    queryFn: ({ signal }) => api.videoProbe(artifactId, signal),
    retry: false,
    staleTime: Infinity,
  });
  const job = useQuery({
    queryKey: ["video-utility-job", jobId],
    queryFn: ({ signal }) => api.videoUtilityJob(jobId ?? "", signal),
    enabled: jobId !== null,
    refetchInterval: (query) => query.state.data && FINISHED.has(query.state.data.status) ? false : 500,
  });
  const working = sending || (jobId !== null && (!job.data || !FINISHED.has(job.data.status)));
  const follow = () => setAt(video.current?.currentTime ?? 0);
  const save = async () => {
    if (working) return;
    setProblem(null);
    setSending(true);
    try {
      const queued = await api.saveVideoFrame(artifactId, video.current?.currentTime ?? at);
      setJobId(queued.id);
    } catch (error) {
      setProblem(error instanceof ApiError || error instanceof Error ? error.message : "The frame was not saved.");
    } finally {
      setSending(false);
    }
  };

  if (probe.isPending) return <p role="status">Reading the video…</p>;
  if (probe.error) return <p role="alert">{probe.error.message}</p>;
  const facts = probe.data;
  if (!facts.can_save_frame) {
    return (
      <div role="alert">
        <p>A frame cannot be saved from this video.</p>
        <ul>{facts.limits.map((limit) => <li key={limit}>{LIMIT_TEXT[limit]}</li>)}</ul>
      </div>
    );
  }
  const result = job.data?.status === "complete" ? job.data.result_json : null;
  const requested = result ? numberField(result, "requested_seconds") : null;
  const actual = result ? numberField(result, "actual_seconds") : null;
  return (
    <div className="video-frame-picker">
      {/* Covered or hidden here as everywhere else the video appears, until someone shows it. */}
      <ShieldedMedia kind="video">
        {/* A stored video has no caption track to point at, and an empty one would claim one. */}
        {/* eslint-disable-next-line jsx-a11y-x/media-has-caption */}
        <video ref={video} src={source} controls preload="metadata" aria-label="Video to take a frame from"
          onTimeUpdate={follow} onSeeked={follow} onLoadedMetadata={follow} />
      </ShieldedMedia>
      <p>
        {facts.video.display_width} × {facts.video.display_height}
        {facts.duration_seconds !== null && <> · {seconds(facts.duration_seconds)} long</>}
        {facts.video.frame_rate_form === "variable" && <> · frames are not evenly spaced</>}
      </p>
      <p>The saved frame is the one shown at {seconds(at)}.</p>
      <button type="button" className="primary" aria-disabled={working} onClick={() => void save()}>
        {working ? "Saving the frame…" : "Save this frame"}
      </button>
      {problem && <p role="alert">{problem}</p>}
      {job.data?.status === "failed" && <p role="alert">{job.data.error ?? "The frame was not saved."}</p>}
      {job.data?.status === "cancelled" && <p role="status">Saving the frame was cancelled.</p>}
      {working && job.data && <p role="status">{job.data.status === "queued" ? "Waiting its turn…" : "Saving the frame…"}</p>}
      {requested !== null && actual !== null && (
        <p role="status">
          Saved the frame at {seconds(actual)}
          {Math.abs(actual - requested) >= 0.0005 && <> (asked for {seconds(requested)})</>}. It is in the Media Library.
        </p>
      )}
    </div>
  );
}

/** A caption button that opens the frame picker for one stored video. */
export function VideoFrameButton({ artifactId, source }: { artifactId: string; source: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className="icon-button" aria-label="Save a frame from this video"
        title="Save a frame" onClick={() => setOpen(true)}>
        <Camera size={14} aria-hidden="true" />
      </button>
      {/* Outside the caption it is opened from, so the caption's own styles
          for its small controls never reach the dialog's buttons. */}
      {open && createPortal(
        <AccessibleDialog title="Save a frame" eyebrow="Video utilities"
          closeLabel="Close save a frame" onClose={() => setOpen(false)} className="video-frame-dialog">
          <VideoFrameBody artifactId={artifactId} source={source} />
        </AccessibleDialog>,
        document.body,
      )}
    </>
  );
}
