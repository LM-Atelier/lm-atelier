import { useId, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation } from "@tanstack/react-query";
import { Scissors } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { ApiError, api } from "./api";
import { ShieldedMedia } from "./ShieldedMedia";
import type { Job, VideoProbe, VideoTrimMode, VideoTrimPreview, VideoTrimRequest } from "./types";
import { EXACT_LIMIT_TEXT, SAME_MOMENT, previewLines, savedLines, workingText } from "./videoTrimText";
import { FINISHED, LIMIT_TEXT, useVideoProbe, useVideoUtilityJob } from "./videoUtilityText";

interface Cut {
  start: number;
  end: number;
  keepAudio: boolean;
  mode: VideoTrimMode;
}

/** A typed time, to the millisecond and within the video, or null while the field holds no number. */
function fieldTime(text: string, duration: number): number | null {
  if (text.trim() === "") return null;
  const value = Number(text);
  if (!Number.isFinite(value)) return null;
  return Math.min(Math.max(Math.round(value * 1000) / 1000, 0), duration);
}

function fieldText(value: number): string {
  return String(Math.round(value * 1000) / 1000);
}

function sameCut(one: Cut, other: Cut): boolean {
  return Math.abs(one.start - other.start) < SAME_MOMENT
    && Math.abs(one.end - other.end) < SAME_MOMENT
    && one.keepAudio === other.keepAudio
    && one.mode === other.mode;
}

function checkedCut(preview: VideoTrimPreview): Cut {
  return {
    start: preview.requested_start_seconds,
    end: preview.requested_end_seconds,
    keepAudio: preview.keep_audio,
    mode: preview.mode,
  };
}

/** The choice between copying a part unchanged and re-encoding it on its exact frames. */
function TrimModeChoice({ mode, facts, onChoose }: {
  mode: VideoTrimMode;
  facts: VideoProbe;
  onChoose: (mode: VideoTrimMode) => void;
}) {
  const name = useId();
  const copyHint = useId();
  const exactHint = useId();
  const offered = facts.can_trim_exact;
  // Only reasons that stop an exact cut; sound it cannot keep is said beside the sound.
  const reasons = facts.exact_trim_limits.filter((limit) => limit !== "exact-trim-audio-unsupported");
  return (
    <div role="radiogroup" aria-label="How to cut">
      <label className="generation-record-choice">
        <input type="radio" name={name} checked={mode === "copy"} aria-describedby={copyHint}
          onChange={() => onChoose("copy")} />
        <span>Keep the original quality</span>
      </label>
      <p className="muted" id={copyHint}>
        Copied without re-encoding, so its quality is unchanged. A copy can only begin on a keyframe, a frame the
        video stores whole, so it may begin a little before your start and run a few frames past your end.
      </p>
      <label className="generation-record-choice">
        <input type="radio" name={name} checked={mode === "exact"} aria-describedby={exactHint}
          aria-disabled={!offered} onChange={() => { if (offered) onChoose("exact"); }} />
        <span>Cut on the exact frames</span>
      </label>
      <p className="muted" id={exactHint}>
        Re-encoded as a new H.264 video, so it begins and ends on the frames you chose. Its picture is close to the
        original but not identical, and it takes longer.
      </p>
      {!offered && (
        <div>
          <p>This video cannot be cut on exact frames.</p>
          <ul>{reasons.map((limit) => <li key={limit}>{EXACT_LIMIT_TEXT[limit]}</li>)}</ul>
        </div>
      )}
    </div>
  );
}

function TrimOutcome({ job }: { job: Job }) {
  if (job.status === "complete") {
    return <div role="status">{savedLines(job.result_json).map((line) => <p key={line}>{line}</p>)}</div>;
  }
  if (job.status === "failed") return <p role="alert">{job.error ?? "The video was not trimmed."}</p>;
  if (job.status === "cancelled") return <p role="status">Trimming was stopped. Nothing was saved.</p>;
  return <p role="status">Trimming was interrupted when the app stopped. It starts again when the app restarts.</p>;
}

/** Choose the part of a stored video to keep and how to cut it, check what the cut would really keep, then make it. */
function VideoTrimEditor({ artifactId, source, facts, duration }: {
  artifactId: string;
  source: string;
  facts: VideoProbe;
  duration: number;
}) {
  // Held as state rather than a ref, so the buttons that read the player know
  // whether it can be played: a hidden video renders none, and a blurred one
  // sits inside an inert cover. Showing either renders a new, uncovered player.
  const [player, setPlayer] = useState<HTMLVideoElement | null>(null);
  const [startText, setStartText] = useState("0");
  const [endText, setEndText] = useState(() => fieldText(duration));
  const [soundChosen, setSoundChosen] = useState(true);
  // A copy unless the person chooses otherwise, each time the dialog opens.
  const [mode, setMode] = useState<VideoTrimMode>("copy");
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useVideoUtilityJob(jobId);
  const stop = useMutation({
    mutationFn: (id: string) => api.cancelJob(id),
    onSuccess: () => void job.refetch(),
  });
  const check = useMutation({
    mutationFn: (cut: Cut) => api.videoTrimPreview(artifactId, cut.start, cut.end, cut.keepAudio, cut.mode),
  });
  const trim = useMutation({
    mutationFn: (body: VideoTrimRequest) => api.trimVideo(artifactId, body),
    onSuccess: (queued) => {
      stop.reset();
      setJobId(queued.id);
    },
    // The preview no longer says where this cut would start, so it goes with
    // the refusal that says so, and the cut has to be checked again.
    onError: (error) => {
      if (error instanceof ApiError && error.code === "video-trim-preview-stale") check.reset();
    },
  });

  const hasSound = facts.audio.length > 0;
  const soundOffered = hasSound && (mode === "exact" ? facts.can_keep_audio_exact : facts.can_keep_audio);
  const start = fieldTime(startText, duration);
  const end = fieldTime(endText, duration);
  const cut: Cut | null = start !== null && end !== null
    ? { start, end, keepAudio: soundOffered && soundChosen, mode }
    : null;
  // A preview speaks only for the cut it was asked about: change the cut and
  // it is gone, and comes back if the cut is changed back.
  const preview = cut && check.data && sameCut(checkedCut(check.data), cut) ? check.data : null;
  const checkProblem = cut && check.variables && sameCut(check.variables, cut) ? check.error : null;
  // While a new trim is being sent, the job still followed is the one before it.
  const posting = trim.isPending;
  const current = posting ? undefined : job.data;
  const working = posting || (jobId !== null && (!current || !FINISHED.has(current.status)));
  const trimmable = preview !== null && !preview.keeps_whole_video && !working;
  const finished = current && FINISHED.has(current.status) ? current : null;
  const playable = player !== null && player.closest("[inert]") === null;

  const fromPlayer = (setText: (text: string) => void) => {
    if (!playable) return;
    setText(fieldText(Math.min(Math.max(player.currentTime, 0), duration)));
  };
  const runCheck = () => {
    if (!cut || check.isPending || working) return;
    trim.reset();
    check.mutate(cut);
  };
  const runTrim = () => {
    if (!cut || !preview || preview.keeps_whole_video || working) return;
    trim.mutate({
      start_seconds: cut.start,
      end_seconds: cut.end,
      keep_audio: cut.keepAudio,
      shown_start_seconds: preview.start_seconds,
      mode: cut.mode,
      // An exact cut is bound to the frames it showed as well as to where it starts.
      ...(cut.mode === "exact" ? { shown_frame_count: preview.frame_count } : {}),
    });
  };
  const stopTrimming = () => {
    if (jobId === null || posting || !working || stop.isPending) return;
    stop.mutate(jobId);
  };

  return (
    <div className="video-utility-body">
      <p>Choose the part to keep, and how to cut it. The original is not changed.</p>
      {/* Covered or hidden here as everywhere else the video appears, until someone shows it. */}
      <ShieldedMedia kind="video">
        {/* A stored video has no caption track to point at, and an empty one would claim one. */}
        {/* eslint-disable-next-line jsx-a11y-x/media-has-caption */}
        <video ref={setPlayer} src={source} controls preload="metadata" aria-label="Video to trim" />
      </ShieldedMedia>
      <label>
        Start (seconds)
        <input type="number" min={0} max={duration} step={0.001} value={startText}
          onChange={(event) => setStartText(event.target.value)}
          onBlur={() => { if (start !== null) setStartText(fieldText(start)); }} />
      </label>
      <label>
        End (seconds)
        <input type="number" min={0} max={duration} step={0.001} value={endText}
          onChange={(event) => setEndText(event.target.value)}
          onBlur={() => { if (end !== null) setEndText(fieldText(end)); }} />
      </label>
      <div className="row-actions">
        <button type="button" className="secondary compact-button" aria-disabled={!playable}
          onClick={() => fromPlayer(setStartText)}>
          Start here
        </button>
        <button type="button" className="secondary compact-button" aria-disabled={!playable}
          onClick={() => fromPlayer(setEndText)}>
          End here
        </button>
      </div>
      <TrimModeChoice mode={mode} facts={facts} onChoose={(chosen) => { if (!working) setMode(chosen); }} />
      {soundOffered && (
        <label className="generation-record-choice">
          <input type="checkbox" checked={soundChosen} onChange={(event) => setSoundChosen(event.target.checked)} />
          <span>Keep the sound</span>
        </label>
      )}
      {hasSound && !soundOffered && (
        <p>
          {mode === "exact"
            ? "Its sound cannot be re-encoded for an exact cut, so the new video has none."
            : "Its sound cannot be copied unchanged, so the new video has none."}
        </p>
      )}
      {!cut && <p>Type the start and the end in seconds.</p>}
      <div className="row-actions">
        <button type="button" className="secondary" aria-disabled={!cut || check.isPending || working} onClick={runCheck}>
          Check the cut
        </button>
        <button type="button" className="primary" aria-disabled={!trimmable} onClick={runTrim}>
          Trim
        </button>
      </div>
      {check.isPending && <p role="status">Checking the cut…</p>}
      {checkProblem && <p role="alert">{checkProblem.message}</p>}
      {preview && (
        <div role="status">{previewLines(preview, facts.can_trim_exact, hasSound).map((line) => <p key={line}>{line}</p>)}</div>
      )}
      {trim.error && <p role="alert">{trim.error.message}</p>}
      {/* Once there is a job the stop button stays, unusable after the job ends,
          since taking it away would drop the focus it may hold out of the dialog. */}
      {(working || jobId !== null) && (
        <div className="row-actions">
          {working && <p role="status">{workingText(current)}</p>}
          {!posting && jobId !== null && (
            <button type="button" className="secondary" aria-disabled={!working || stop.isPending} onClick={stopTrimming}>
              Stop trimming
            </button>
          )}
        </div>
      )}
      {working && stop.error && (
        <p role="alert">
          {stop.error instanceof ApiError && stop.error.status === 409
            ? "It was already being saved, so it will finish."
            : stop.error.message}
        </p>
      )}
      {finished && <TrimOutcome job={finished} />}
    </div>
  );
}

/** Read the video first: whether it can be trimmed, and how long it is. */
function VideoTrimBody({ artifactId, source }: { artifactId: string; source: string }) {
  const probe = useVideoProbe(artifactId);
  if (probe.isPending) return <p role="status">Reading the video…</p>;
  if (probe.error) return <p role="alert">{probe.error.message}</p>;
  const facts = probe.data;
  if (!facts.can_trim || facts.duration_seconds === null) {
    return (
      <div role="alert">
        <p>This video cannot be trimmed.</p>
        <ul>{facts.limits.map((limit) => <li key={limit}>{LIMIT_TEXT[limit]}</li>)}</ul>
      </div>
    );
  }
  return <VideoTrimEditor artifactId={artifactId} source={source} facts={facts} duration={facts.duration_seconds} />;
}

/** A caption button that opens the trim dialog for one stored video. */
export function VideoTrimButton({ artifactId, source }: { artifactId: string; source: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className="icon-button" aria-label="Trim this video"
        title="Trim" onClick={() => setOpen(true)}>
        <Scissors size={14} aria-hidden="true" />
      </button>
      {/* Outside the caption it is opened from, so the caption's own styles
          for its small controls never reach the dialog's buttons. */}
      {open && createPortal(
        <AccessibleDialog title="Trim a video" eyebrow="Video utilities"
          closeLabel="Close trim a video" onClose={() => setOpen(false)} className="video-utility-dialog">
          <VideoTrimBody artifactId={artifactId} source={source} />
        </AccessibleDialog>,
        document.body,
      )}
    </>
  );
}
