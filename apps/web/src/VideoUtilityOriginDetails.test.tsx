import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { ArtifactGenerationDetails } from "./ArtifactGenerationDetails";
import { SENSITIVE_MEDIA_KEY } from "./sensitiveMedia";
import type { Artifact } from "./types";
import { videoUtilityOrigin } from "./videoUtilityText";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: { artifact: vi.fn(), run: vi.fn() } };
});

const stamp = "2026-10-06T12:00:00Z";
const resultId = `sha256:${"b".repeat(64)}`;
const sourceId = `sha256:${"a".repeat(64)}`;
const TRIM = {
  action: "trim", source_artifact_id: sourceId, source_sha256: "a".repeat(64),
  requested_start_seconds: 1.43, requested_end_seconds: 2.7, actual_start_seconds: 1,
  keyframe_seconds: 1, actual_end_seconds: 2.9, from_beginning: false,
};
const EXACT_TRIM = {
  ...TRIM, mode: "exact", actual_start_seconds: 1.4, actual_end_seconds: 2.7,
  frames: { first_seconds: 1.4, last_seconds: 2.6, count: 13 },
};
const FRAME = {
  action: "extract_frame", source_artifact_id: sourceId, source_sha256: "a".repeat(64),
  requested_seconds: 0.43, actual_seconds: 0.4,
};

function artifact(id: string, name: string, metadata: Record<string, unknown>): Artifact {
  return {
    id, sha256: id.slice(7), kind: "video", media_type: "video/mp4", size_bytes: 1024,
    original_name: name, metadata_json: metadata, created_at: stamp,
  };
}

async function details(metadata: Record<string, unknown>, source: Artifact | Error) {
  vi.mocked(api.artifact).mockImplementation(async (id: string) => {
    if (id === resultId) return artifact(resultId, "walk trim 1.000-2.900s.mp4", metadata);
    if (source instanceof Error) throw source;
    return source;
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><ArtifactGenerationDetails artifactId={resultId} /></QueryClientProvider>);
  const summary = screen.getByText("Generation details", { selector: "summary" });
  const disclosure = summary.parentElement as HTMLDetailsElement;
  disclosure.open = true;
  fireEvent(disclosure, new Event("toggle"));
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  localStorage.removeItem(SENSITIVE_MEDIA_KEY);
});

describe("the video a saved frame or a trim came from", () => {
  it("names the source and the part a trim kept beside the part asked for", async () => {
    await details({ video_trim: TRIM }, artifact(sourceId, "walk.mp4", {}));

    expect(await screen.findByText("Trimmed from walk.mp4, copied without re-encoding.")).toBeVisible();
    expect(screen.getByText(/The part from 1\.000 s to 2\.900 s \(asked for 1\.430 s to 2\.700 s\)\./)).toBeVisible();
    expect(screen.queryByText("Model and workflow names were not recorded.")).toBeNull();
    expect(api.run).not.toHaveBeenCalled();
  });

  it("says an exact cut was re-encoded, and names the frames it kept", async () => {
    await details({ video_trim: EXACT_TRIM }, artifact(sourceId, "walk.mp4", {}));

    expect(await screen.findByText("Trimmed from walk.mp4, re-encoded to start and end on the chosen frames.")).toBeVisible();
    expect(screen.getByText(/The 13 frames from 1\.400 s to 2\.700 s \(asked for 1\.430 s to 2\.700 s\)\./)).toBeVisible();
    expect(screen.queryByText(/without re-encoding/)).toBeNull();
  });

  it("names the frame a picture holds and the time that was asked for", async () => {
    await details({ video_frame: FRAME }, artifact(sourceId, "walk.mp4", {}));

    expect(await screen.findByText("Saved as a picture from walk.mp4.")).toBeVisible();
    expect(screen.getByText("The frame at 0.400 s (asked for 0.430 s).")).toBeVisible();
  });

  it("says plainly when the source is no longer stored", async () => {
    await details({ video_trim: { ...TRIM, from_beginning: true } }, new ApiError(404, "gone", "artifact not found"));

    expect(await screen.findByText("Trimmed from a video that is no longer stored, copied without re-encoding.")).toBeVisible();
    expect(screen.getByText(/The part from the beginning to 2\.900 s/)).toBeVisible();
  });

  it("covers the source's name while media is covered", async () => {
    localStorage.setItem(SENSITIVE_MEDIA_KEY, "blur");
    await details({ video_frame: FRAME }, artifact(sourceId, "walk.mp4", {}));

    expect(await screen.findByText("Saved as a picture from a stored video.")).toBeVisible();
    expect(screen.queryByText(/walk\.mp4/)).toBeNull();
  });

  it("reports a source that could not be read rather than calling it gone", async () => {
    await details({ video_frame: FRAME }, new ApiError(503, "busy", "unavailable"));

    expect(await screen.findByRole("alert")).toHaveTextContent("Generation details could not be loaded.");
  });
});

describe("reading a video utility's record from stored metadata", () => {
  it("reads a frame and a trim and nothing else", () => {
    expect(videoUtilityOrigin({ video_frame: FRAME })).toEqual({
      action: "extract_frame", sourceId, requested: 0.43, actual: 0.4,
    });
    expect(videoUtilityOrigin({ video_trim: TRIM })).toEqual({
      action: "trim", sourceId, requestedStart: 1.43, requestedEnd: 2.7, actualStart: 1, actualEnd: 2.9,
      fromBeginning: false, exactFrames: null,
    });
    expect(videoUtilityOrigin({ video_trim: EXACT_TRIM })).toMatchObject({ actualStart: 1.4, exactFrames: 13 });
    // A copy's record names no frames, whatever else it holds.
    expect(videoUtilityOrigin({ video_trim: { ...TRIM, frames: { count: 13 } } })).toMatchObject({ exactFrames: null });
    expect(videoUtilityOrigin({ run_id: "run" })).toBeNull();
    expect(videoUtilityOrigin({ video_trim: { ...TRIM, actual_end_seconds: "late" } })).toBeNull();
    expect(videoUtilityOrigin({ video_frame: { ...FRAME, source_artifact_id: 7 } })).toBeNull();
  });
});
