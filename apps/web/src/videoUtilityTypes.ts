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
