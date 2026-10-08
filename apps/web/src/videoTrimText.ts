import type { ExactTrimLimit, Job, VideoTrimPreview, VideoTrimResult } from "./types";
import { numberField, seconds } from "./videoUtilityText";

/** The stages a running trim names, each shown as it is. */
const STAGES = new Set([
  "Reading the video",
  "Finding the keyframe",
  "Copying the chosen part",
  "Finding the frames",
  "Re-encoding the chosen part",
  "Checking the new video",
  "Comparing it with the original",
]);
const FORMAT_NAMES: Record<VideoTrimPreview["format"], string> = { mp4: "MP4", webm: "WebM", matroska: "Matroska" };
/** Closer than this, two times are the same moment at the millisecond the dialog shows. */
export const SAME_MOMENT = 0.0005;
/** Picture and sound this far apart are worth mentioning; closer than this, nobody would notice. */
const NOTICEABLE_GAP = 0.01;

/** Why a video cannot be cut on exact frames, as the person is told it. */
export const EXACT_LIMIT_TEXT: Record<ExactTrimLimit, string> = {
  "exact-trim-picture-unsupported":
    "Its picture is stored in a form an exact cut cannot re-encode without changing how it looks, such as 10-bit, HDR, full-range or interlaced video.",
  "exact-trim-frame-size-unsupported": "Its frames have an odd width or height, or are larger than an exact cut re-encodes.",
  "exact-trim-turned-unsupported": "It is turned, and an exact cut cannot keep the turn yet.",
  "exact-trim-audio-unsupported": "Its sound cannot be re-encoded for an exact cut, so an exact cut leaves it out.",
};

/** Where a checked copy really starts, measured against the start that was asked for. */
function copyStartLine(preview: VideoTrimPreview): string {
  const asked = preview.requested_start_seconds;
  const keyframe = preview.keyframe_seconds;
  if (preview.from_beginning) {
    return asked <= SAME_MOMENT
      ? "It starts at the beginning."
      : `It starts at the beginning of the video, ${seconds(asked)} before your start, because your start comes before its second keyframe.`;
  }
  if (Math.abs(keyframe - asked) < SAME_MOMENT) return `It starts at ${seconds(keyframe)}, on a keyframe.`;
  return `It starts at the keyframe at ${seconds(keyframe)}, ${seconds(asked - keyframe)} before your start.`;
}

/** Where a checked exact cut starts: on the frame on screen at the start asked for. */
function exactStartLine(preview: VideoTrimPreview): string {
  const first = preview.start_seconds;
  if (preview.began_at_first_picture) {
    return `Your start comes before the video's first picture, so it starts on that picture, at ${seconds(first)}.`
      + (preview.audio_streams_kept > 0 ? " The sound before it is left out." : "");
  }
  if (Math.abs(first - preview.requested_start_seconds) < SAME_MOMENT) {
    return `It starts on the frame at ${seconds(first)}, your start.`;
  }
  return `It starts on the frame shown at your start, which begins at ${seconds(first)}.`;
}

function copySoundLine(kept: number): string {
  if (kept === 0) return "Without sound.";
  return kept === 1 ? "It keeps its sound." : `It keeps its ${kept} sound tracks.`;
}

function exactSoundLine(kept: number): string {
  if (kept === 0) return "Without sound.";
  return kept === 1
    ? "Its sound is re-encoded as AAC and cut with the picture."
    : `Its ${kept} sound tracks are re-encoded as AAC and cut with the picture.`;
}

function exactLines(preview: VideoTrimPreview, hasSound: boolean): string[] {
  const lines = [exactStartLine(preview)];
  const last = preview.last_frame_seconds;
  const end = preview.end_seconds;
  const count = preview.frame_count;
  if (last !== null && end !== null && count !== null) {
    lines.push(
      `It ends after the frame at ${seconds(last)}, the last one shown before your end: ${count} ${count === 1 ? "frame" : "frames"}, ${seconds(end - preview.start_seconds)} in all.`,
    );
  }
  lines.push(
    "Every frame is re-encoded as H.264, so its picture is close to the original but not identical, and it takes longer than a copy. How close is measured once it is made.",
    exactSoundLine(preview.audio_streams_kept),
    "Saved as a new MP4 video. The original is not changed.",
  );
  if (preview.omitted_streams > 0) lines.push("Subtitles, cover pictures and other streams are left out.");
  if (preview.keeps_whole_video) {
    // Without its sound, a copy of the whole video is a change; a silent video has no such copy.
    lines.push("That keeps every frame of the video. Choose a later start or an earlier end."
      + (hasSound ? " To leave out only its sound, keep the original quality instead." : ""));
  }
  return lines;
}

/** What a checked cut would keep, a sentence at a time.
 *
 * `exactOffered` says whether the same part could be cut on its exact frames
 * instead, so a copy that starts early can say so; `hasSound` whether the video has any.
 */
export function previewLines(preview: VideoTrimPreview, exactOffered: boolean, hasSound: boolean): string[] {
  if (preview.mode === "exact") return exactLines(preview, hasSound);
  const lines = [
    copyStartLine(preview),
    `It ends near ${seconds(preview.requested_end_seconds)}. Its real end is measured once it is made.`,
    `Saved as a new ${FORMAT_NAMES[preview.format]} video. The original is not changed.`,
    copySoundLine(preview.audio_streams_kept),
  ];
  if (exactOffered && preview.requested_start_seconds - preview.start_seconds >= 2 * SAME_MOMENT) {
    lines.push("To start on the frame at your start instead, choose Cut on the exact frames and check again.");
  }
  if (preview.omitted_streams > 0) lines.push("Subtitles, cover pictures and other streams are left out.");
  if (preview.format === "matroska") lines.push("Some browsers cannot play Matroska videos; it can still be downloaded.");
  if (preview.keeps_whole_video) {
    // Leaving out the sound changes the video only when the cut would keep some.
    lines.push(preview.audio_streams_kept > 0
      ? "That would keep the whole video as it is. Choose an earlier end, or leave out the sound."
      : "That would keep the whole video as it is. Choose an earlier end.");
  }
  return lines;
}

/** How close an exact cut measured to the original, or nothing when its result does not say. */
function closenessLine(result: Record<string, unknown>): string | null {
  const quality = result.quality;
  const ssim = numberField(quality, "ssim_mean");
  if (ssim === null || typeof quality !== "object" || quality === null) return null;
  const lowest = (quality as Record<string, unknown>).psnr_lowest_db;
  if (lowest === null) return "Every frame decoded identical to the original's.";
  if (typeof lowest !== "number" || !Number.isFinite(lowest)) return null;
  return `Measured against the original frame by frame: SSIM ${ssim.toFixed(3)} on average (1 is identical), and no frame below ${lowest.toFixed(1)} dB PSNR.`;
}

/** What a finished trim made, beside what was chosen, read from its job's result. */
export function savedLines(result: Record<string, unknown>): string[] {
  const field = (key: keyof VideoTrimResult) => numberField(result, key);
  const start = field("actual_start_seconds");
  const end = field("actual_end_seconds");
  const askedStart = field("requested_start_seconds");
  const askedEnd = field("requested_end_seconds");
  if (start === null || end === null || askedStart === null || askedEnd === null) return ["Saved a new video."];
  const inLibrary = result.in_library !== false;
  const chosen = `(you chose ${seconds(askedStart)} to ${seconds(askedEnd)})`;
  const count = numberField(result.frames, "count");
  const saved = result.mode === "exact" && count !== null
    ? `Saved a new video of ${count} ${count === 1 ? "frame" : "frames"}, from ${seconds(start)} to ${seconds(end)} of the original ${chosen}.`
    : `Saved a new video from ${result.from_beginning === true ? "the beginning" : seconds(start)} to ${seconds(end)} of the original ${chosen}.`;
  const lines = [saved + (inLibrary ? " It is in the Media Library." : "")];
  if (!inLibrary) {
    lines.push("The same video is already in Recently Deleted. Restore it there to see it in the Media Library.");
  }
  const closeness = result.mode === "exact" ? closenessLine(result) : null;
  if (closeness) lines.push(closeness);
  const audio = result.audio;
  if (result.keep_audio === true && Array.isArray(audio)) {
    const picture = numberField(result.video, "start_seconds");
    const sound = numberField(audio[0], "start_seconds");
    if (picture !== null && sound !== null && Math.abs(picture - sound) >= NOTICEABLE_GAP) {
      lines.push(`Its picture begins ${seconds(Math.abs(picture - sound))} ${picture > sound ? "after" : "before"} its sound.`);
    }
  }
  return lines;
}

export function workingText(job: Job | undefined): string {
  if (!job) return "Trimming…";
  if (job.status === "queued" || job.status === "paused") return "Waiting its turn…";
  return STAGES.has(job.phase) ? `${job.phase}…` : "Trimming…";
}
