import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { JobStatus, VideoUtilityLimit } from "./types";

/** Why a video utility is not offered for a video, as the person is told it. */
export const LIMIT_TEXT: Record<VideoUtilityLimit, string> = {
  "video-codec-unsupported": "Its video is stored in a format the video utilities do not decode.",
  "audio-codec-unsupported": "Its sound cannot be copied unchanged.",
  "video-duration-unknown": "It does not say how long it is.",
  "video-duration-too-long": "It is longer than the video utilities read.",
  "video-frame-too-large": "Its frames are larger than the video utilities read.",
  "video-rotation-unsupported": "It is turned by an angle other than a quarter turn.",
};

/** The states after which nothing more happens to a job. */
export const FINISHED: ReadonlySet<JobStatus> = new Set<JobStatus>(["complete", "failed", "cancelled", "interrupted"]);

/** Writes a moment in a video to the millisecond. */
export function seconds(value: number): string {
  return value.toFixed(3) + " s";
}

/** The finite number held under a key of a job's result, or null when it holds none. */
export function numberField(record: unknown, key: string): number | null {
  if (typeof record !== "object" || record === null) return null;
  const value = (record as Record<string, unknown>)[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** What the video utilities read about one stored video. Read once, and shared by every dialog that asks. */
export function useVideoProbe(artifactId: string) {
  return useQuery({
    queryKey: ["video-probe", artifactId],
    queryFn: ({ signal }) => api.videoProbe(artifactId, signal),
    retry: false,
    staleTime: Infinity,
  });
}

/**
 * A video utility job, read again every half second until it finishes.
 *
 * A read that fails stops the reading until someone asks again, rather than
 * repeating twice a second for as long as the dialog stays open.
 */
export function useVideoUtilityJob(jobId: string | null) {
  return useQuery({
    queryKey: ["video-utility-job", jobId],
    queryFn: ({ signal }) => api.videoUtilityJob(jobId ?? "", signal),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      if (query.state.status === "error") return false;
      return query.state.data && FINISHED.has(query.state.data.status) ? false : 500;
    },
  });
}

/** Said in each video utility while the tools it runs are builds found on the computer. */
export const FOUND_TOOLS_TEXT =
  "The video utilities use the ffmpeg and ffprobe found on this computer, which the app has not checked.";

/** The build of a media tool that made something, as its record names it. */
export interface ToolBuild {
  name: string;
  version: string;
  /** Found on this computer rather than provided and checked by the app. */
  found: boolean;
}

/** The tool build a record names, or null when it names none. */
export function toolBuild(record: unknown): ToolBuild | null {
  if (typeof record !== "object" || record === null) return null;
  const { name, version, origin } = record as Record<string, unknown>;
  if (typeof name !== "string" || typeof version !== "string" || typeof origin !== "string") return null;
  return { name, version, found: origin === "system" };
}

/** Says which build made something, and that a build found on the computer is not one the app checked. */
export function madeWithText(tool: ToolBuild): string {
  const build = `${tool.name} ${tool.version}`;
  return tool.found ? `Made with ${build} found on this computer, which the app has not checked.` : `Made with ${build}.`;
}

/** What a picture or video made by a video utility records about the video it came from. */
export type VideoUtilityOrigin =
  | { action: "extract_frame"; sourceId: string; requested: number; actual: number; madeWith: ToolBuild | null }
  | {
    action: "trim";
    sourceId: string;
    requestedStart: number;
    requestedEnd: number;
    actualStart: number;
    actualEnd: number;
    fromBeginning: boolean;
    /** How many frames an exact cut kept; null for a copy, which records none. */
    exactFrames: number | null;
    madeWith: ToolBuild | null;
  };

/** The origin a stored file's metadata records, or null when it was not made by a video utility. */
export function videoUtilityOrigin(metadata: Record<string, unknown>): VideoUtilityOrigin | null {
  const frame = metadata.video_frame;
  if (typeof frame === "object" && frame !== null) {
    const record = frame as Record<string, unknown>;
    const requested = numberField(record, "requested_seconds");
    const actual = numberField(record, "actual_seconds");
    if (typeof record.source_artifact_id === "string" && requested !== null && actual !== null) {
      return {
        action: "extract_frame", sourceId: record.source_artifact_id, requested, actual, madeWith: toolBuild(record.tool),
      };
    }
  }
  const trim = metadata.video_trim;
  if (typeof trim === "object" && trim !== null) {
    const record = trim as Record<string, unknown>;
    const values = ["requested_start_seconds", "requested_end_seconds", "actual_start_seconds", "actual_end_seconds"]
      .map((key) => numberField(record, key));
    const [requestedStart, requestedEnd, actualStart, actualEnd] = values;
    if (
      typeof record.source_artifact_id === "string"
      && requestedStart !== null && requestedEnd !== null && actualStart !== null && actualEnd !== null
    ) {
      return {
        action: "trim", sourceId: record.source_artifact_id, requestedStart, requestedEnd, actualStart, actualEnd,
        fromBeginning: record.from_beginning === true,
        // A record with no mode was made before cuts had one, so it is a copy.
        exactFrames: record.mode === "exact" ? numberField(record.frames, "count") : null,
        madeWith: toolBuild(record.tool),
      };
    }
  }
  return null;
}
