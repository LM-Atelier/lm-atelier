export type VideoUtilityLimit =
  | "video-codec-unsupported"
  | "audio-codec-unsupported"
  | "video-duration-unknown"
  | "video-duration-too-long"
  | "video-frame-too-large"
  | "video-rotation-unsupported";

/** Why a video cannot be cut on exact frames; the last only leaves its sound out. */
export type ExactTrimLimit =
  | "exact-trim-picture-unsupported"
  | "exact-trim-frame-size-unsupported"
  | "exact-trim-turned-unsupported"
  | "exact-trim-audio-unsupported";

export interface VideoStreamFacts {
  index: number;
  codec: string;
  width: number;
  height: number;
  display_width: number;
  display_height: number;
  sample_aspect: string | null;
  rotation: 0 | 90 | 180 | 270;
  frame_rate: string | null;
  frame_rate_form: "constant" | "variable" | "unknown";
  time_base: string | null;
  /** How the stream states its pictures are stored, by ffmpeg's names; null when it does not say. */
  pixel_format: string | null;
  color_range: string | null;
  color_space: string | null;
  color_transfer: string | null;
  color_primaries: string | null;
  field_order: string | null;
}

export interface AudioStreamFacts {
  index: number;
  codec: string;
  channels: number | null;
  sample_rate: number | null;
}

/** One stored video as the video utilities see it, and what they may do with it. */
export interface VideoProbe {
  version: 1;
  artifact_id: string;
  artifact_sha256: string;
  container: "mp4" | "matroska";
  duration_seconds: number | null;
  start_seconds: number;
  video: VideoStreamFacts;
  audio: AudioStreamFacts[];
  omitted_streams: number;
  can_save_frame: boolean;
  can_trim: boolean;
  can_keep_audio: boolean;
  limits: VideoUtilityLimit[];
  /** Whether a part can be re-encoded to begin and end on exact frames, and keep its sound doing so. */
  can_trim_exact: boolean;
  can_keep_audio_exact: boolean;
  exact_trim_limits: ExactTrimLimit[];
  tool: Record<string, string>;
}

export type VideoTrimFormat = "mp4" | "webm" | "matroska";
export type VideoTrimMediaType = "video/mp4" | "video/webm" | "video/x-matroska";
/** A copy keeps the picture unchanged from a keyframe; an exact cut re-encodes it from the frames chosen. */
export type VideoTrimMode = "copy" | "exact";

/** What a cut of one stored video would keep, read before anything is queued.
 *
 * A copy has no end: it runs to the end of the frames it needs, so its end is
 * measured only once the new video is made. An exact cut names its frames.
 */
export interface VideoTrimPreview {
  version: 1;
  mode: VideoTrimMode;
  artifact_id: string;
  artifact_sha256: string;
  requested_start_seconds: number;
  requested_end_seconds: number;
  keep_audio: boolean;
  /** Where the new video begins on the original's timeline: 0 from the beginning, else the keyframe,
   * and for an exact cut the time of its first frame. */
  start_seconds: number;
  /** The keyframe a copy's picture begins with, or an exact cut is decoded from. */
  keyframe_seconds: number;
  from_beginning: boolean;
  /** For an exact cut: its last frame's time, where the part ends, and how many frames it holds. */
  last_frame_seconds: number | null;
  end_seconds: number | null;
  frame_count: number | null;
  began_at_first_picture: boolean;
  keeps_whole_video: boolean;
  audio_streams_kept: number;
  omitted_streams: number;
  format: VideoTrimFormat;
  media_type: VideoTrimMediaType;
}

/** A cut to make, with the start its preview showed so a changed start is caught,
 * and for an exact cut how many frames it showed. */
export interface VideoTrimRequest {
  start_seconds: number;
  end_seconds: number;
  keep_audio: boolean;
  shown_start_seconds: number;
  mode?: VideoTrimMode;
  shown_frame_count?: number | null;
}

/** The new video's picture, as measured once it was made, on the new video's own timeline. */
export interface TrimmedVideoStream {
  codec: string;
  start_seconds: number | null;
  first_shown_seconds: number;
  end_seconds: number;
  frames_stored: number;
  display_width: number;
  display_height: number;
  rotation: 0 | 90 | 180 | 270;
  sample_aspect: string | null;
}

/** One sound track of the new video, as measured once it was made, on the new video's own timeline. */
export interface TrimmedAudioStream {
  codec: string;
  start_seconds: number | null;
  packets: number;
}

/** The frames an exact cut kept, on the original's timeline. */
export interface ExactTrimFrames {
  first_seconds: number;
  last_seconds: number;
  count: number;
  began_at_first_picture: boolean;
  source_frames_sha256: string;
  /** Where the encode's decode and the check's began, as each decode stated. */
  decoded_from_seconds: number;
  checked_from_seconds: number;
}

/** How close an exact cut's frames measured to the original's; a null PSNR means identical. */
export interface ExactTrimQuality {
  ssim_mean: number;
  psnr_mean_db: number | null;
  psnr_lowest_db: number | null;
}

/** How an exact cut was re-encoded. */
export interface ExactTrimEncoding {
  video_encoder: string;
  preset: string;
  crf: number;
  pixel_format: string;
  audio_encoder: string | null;
  audio_bitrates: number[];
}

/** What a finished trim made, as its job's result holds it.
 *
 * The requested and actual start and end are on the original's timeline. The
 * video and audio entries are measured on the new video's own timeline. A
 * finished job names the new video and its source without holding either, so
 * deleting them frees their space.
 */
export interface VideoTrimResult {
  made_artifact_id: string;
  in_library: boolean;
  action: "trim";
  mode: VideoTrimMode;
  from_artifact_id: string;
  source_sha256: string;
  requested_start_seconds: number;
  requested_end_seconds: number;
  actual_start_seconds: number;
  keyframe_seconds: number;
  actual_end_seconds: number;
  from_beginning: boolean;
  keep_audio: boolean;
  format: VideoTrimFormat;
  media_type: VideoTrimMediaType;
  video: TrimmedVideoStream;
  audio: TrimmedAudioStream[];
  omitted_streams: number;
  /** Only an exact cut records these; a copy records null. */
  frames: ExactTrimFrames | null;
  quality: ExactTrimQuality | null;
  encoding: ExactTrimEncoding | null;
  tool: Record<string, string>;
  measured_with: Record<string, string>;
}
