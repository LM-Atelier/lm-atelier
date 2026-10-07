import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { SENSITIVE_MEDIA_KEY } from "./sensitiveMedia";
import type { Job, VideoProbe } from "./types";
import { VideoFrameButton } from "./VideoFrameButton";

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: { videoProbe: vi.fn(), saveVideoFrame: vi.fn(), videoUtilityJob: vi.fn() },
}));

const clients: QueryClient[] = [];
const PROBE: VideoProbe = {
  version: 1, artifact_id: "sha256:clip", artifact_sha256: "a".repeat(64), container: "mp4",
  duration_seconds: 2, start_seconds: 0,
  video: {
    index: 0, codec: "h264", width: 64, height: 48, display_width: 64, display_height: 48,
    sample_aspect: null, rotation: 0, frame_rate: "10/1", frame_rate_form: "constant", time_base: "1/10240",
  },
  audio: [], omitted_streams: 0, can_save_frame: true, can_trim: true, can_keep_audio: true, limits: [],
  tool: { name: "ffprobe", version: "8.1.2", sha256: "b".repeat(64), origin: "system" },
};
const STAMP = "2026-10-06T00:00:00Z";

function job(status: Job["status"], result_json: Record<string, unknown> = {}, error: string | null = null): Job {
  return {
    id: "job-frame", kind: "media_utility", status, run_id: null, progress: 0, phase: status,
    payload_json: {}, result_json, error, attempt: 1, cancellable: true, created_at: STAMP,
    updated_at: STAMP, started_at: null, completed_at: null,
  } as Job;
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.videoProbe).mockResolvedValue(PROBE);
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  localStorage.clear();
});

function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}>
    <VideoFrameButton artifactId="sha256:clip" source="/api/artifacts/sha256:clip/content" />
  </QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Save a frame from this video" }));
}

function seek(seconds: number) {
  const video = screen.getByLabelText("Video to take a frame from") as HTMLVideoElement;
  Object.defineProperty(video, "currentTime", { configurable: true, value: seconds });
  fireEvent.seeked(video);
}

it("saves the frame at the chosen moment and says which frame it really was", async () => {
  vi.mocked(api.saveVideoFrame).mockResolvedValue(job("queued"));
  vi.mocked(api.videoUtilityJob)
    .mockResolvedValueOnce(job("running"))
    .mockResolvedValue(job("complete", { requested_seconds: 0.43, actual_seconds: 0.4 }));
  open();
  expect(await screen.findByText(/64 × 48/)).toBeInTheDocument();

  seek(0.43);
  expect(screen.getByText("The saved frame is the one shown at 0.430 s.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Save this frame" }));

  expect(await screen.findByText(/Saved the frame at 0.400 s \(asked for 0.430 s\)/)).toBeInTheDocument();
  expect(api.saveVideoFrame).toHaveBeenCalledWith("sha256:clip", 0.43);
  expect(screen.getByRole("button", { name: "Save this frame" })).toHaveAttribute("aria-disabled", "false");
});

it("points to Recently Deleted when the same picture is already there", async () => {
  vi.mocked(api.saveVideoFrame).mockResolvedValue(job("queued"));
  vi.mocked(api.videoUtilityJob).mockResolvedValue(
    job("complete", { requested_seconds: 0.5, actual_seconds: 0.5, in_library: false }),
  );
  open();
  await screen.findByText(/64 × 48/);
  seek(0.5);
  fireEvent.click(screen.getByRole("button", { name: "Save this frame" }));

  expect(await screen.findByText(
    "Saved the frame at 0.500 s. The same picture is already in Recently Deleted. Restore it there to see it in the Media Library.",
  )).toBeInTheDocument();
  expect(screen.queryByText(/It is in the Media Library/)).not.toBeInTheDocument();
  expect(api.saveVideoFrame).toHaveBeenCalledWith("sha256:clip", 0.5);
});

it("keeps the video covered by the same choice as everywhere else it appears", async () => {
  localStorage.setItem(SENSITIVE_MEDIA_KEY, "hide");
  open();
  await screen.findByText(/64 × 48/);

  expect(screen.getByText("This video is hidden.")).toBeInTheDocument();
  expect(screen.queryByLabelText("Video to take a frame from")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Show video" }));
  expect(screen.getByLabelText("Video to take a frame from")).toBeInTheDocument();
});

it("names why a frame cannot be saved instead of offering it", async () => {
  vi.mocked(api.videoProbe).mockResolvedValue({
    ...PROBE, can_save_frame: false, can_trim: false,
    limits: ["video-codec-unsupported", "video-duration-unknown"],
  });
  open();

  expect(await screen.findByText("A frame cannot be saved from this video.")).toBeInTheDocument();
  expect(screen.getByText("Its video is stored in a format the video utilities do not decode.")).toBeInTheDocument();
  expect(screen.getByText("It does not say how long it is.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Save this frame" })).not.toBeInTheDocument();
});

it("shows the server's reason when the frame is refused or the job fails", async () => {
  vi.mocked(api.saveVideoFrame).mockRejectedValueOnce(
    new ApiError(422, "That time is not within this video.", "That time is not within this video.", "video-frame-time-outside"),
  );
  open();
  await screen.findByText(/64 × 48/);

  fireEvent.click(screen.getByRole("button", { name: "Save this frame" }));
  expect(await screen.findByText("That time is not within this video.")).toBeInTheDocument();

  vi.mocked(api.saveVideoFrame).mockResolvedValue(job("queued"));
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("failed", {}, "No frame could be decoded at that time."));
  fireEvent.click(screen.getByRole("button", { name: "Save this frame" }));
  expect(await screen.findByText("No frame could be decoded at that time.")).toBeInTheDocument();
});

it("reports a video that could not be read at all", async () => {
  vi.mocked(api.videoProbe).mockRejectedValue(
    new ApiError(422, "This video could not be read as an MP4 or Matroska file.",
      "This video could not be read as an MP4 or Matroska file.", "video-probe-unreadable"),
  );
  open();

  expect(await screen.findByRole("alert")).toHaveTextContent("This video could not be read as an MP4 or Matroska file.");
});
