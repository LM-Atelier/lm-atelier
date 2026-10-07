export type VideoUtilityLimit =
  | "video-codec-unsupported"
  | "audio-codec-unsupported"
  | "video-duration-unknown"
  | "video-duration-too-long"
  | "video-frame-too-large"
  | "video-rotation-unsupported";

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
  tool: Record<string, string>;
}

export type VideoTrimFormat = "mp4" | "webm" | "matroska";
export type VideoTrimMediaType = "video/mp4" | "video/webm" | "video/x-matroska";

/** What a cut of one stored video would keep, read before anything is queued.
 *
 * It has no end: a copy runs to the end of the frames it needs, so the end is
 * measured only once the new video is made.
 */
export interface VideoTrimPreview {
  version: 1;
  artifact_id: string;
  artifact_sha256: string;
  requested_start_seconds: number;
  requested_end_seconds: number;
  keep_audio: boolean;
  /** Where the new video begins on the original's timeline: 0 from the beginning, else the keyframe. */
  start_seconds: number;
  /** The keyframe the new video's picture begins with. */
  keyframe_seconds: number;
  from_beginning: boolean;
  keeps_whole_video: boolean;
  audio_streams_kept: number;
  omitted_streams: number;
  format: VideoTrimFormat;
  media_type: VideoTrimMediaType;
}

/** A cut to make, with the start its preview showed so a changed start is caught. */
export interface VideoTrimRequest {
  start_seconds: number;
  end_seconds: number;
  keep_audio: boolean;
  shown_start_seconds: number;
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

/** What a finished trim made, as its job's result holds it.
 *
 * The requested and actual start and end are on the original's timeline. The
 * video and audio entries are measured on the new video's own timeline.
 */
export interface VideoTrimResult {
  result_artifact_id: string;
  in_library: boolean;
  action: "trim";
  source_artifact_id: string;
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
  tool: Record<string, string>;
  measured_with: Record<string, string>;
}
