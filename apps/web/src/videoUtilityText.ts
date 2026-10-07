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

/** A video utility job, read again every half second until it finishes. */
export function useVideoUtilityJob(jobId: string | null) {
  return useQuery({
    queryKey: ["video-utility-job", jobId],
    queryFn: ({ signal }) => api.videoUtilityJob(jobId ?? "", signal),
    enabled: jobId !== null,
    refetchInterval: (query) => query.state.data && FINISHED.has(query.state.data.status) ? false : 500,
  });
}
