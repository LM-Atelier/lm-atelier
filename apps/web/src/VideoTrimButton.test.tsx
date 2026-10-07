import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { SENSITIVE_MEDIA_KEY } from "./sensitiveMedia";
import type { Job, VideoProbe, VideoTrimPreview, VideoTrimResult } from "./types";
import { VideoTrimButton } from "./VideoTrimButton";

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: {
    videoProbe: vi.fn(), videoTrimPreview: vi.fn(), trimVideo: vi.fn(), videoUtilityJob: vi.fn(), cancelJob: vi.fn(),
  },
}));

const clients: QueryClient[] = [];
const PROBE: VideoProbe = {
  version: 1, artifact_id: "sha256:clip", artifact_sha256: "a".repeat(64), container: "mp4",
  duration_seconds: 2, start_seconds: 0,
  video: {
    index: 0, codec: "h264", width: 64, height: 48, display_width: 64, display_height: 48,
    sample_aspect: null, rotation: 0, frame_rate: "10/1", frame_rate_form: "constant", time_base: "1/10240",
  },
  audio: [{ index: 1, codec: "aac", channels: 2, sample_rate: 48000 }],
  omitted_streams: 0, can_save_frame: true, can_trim: true, can_keep_audio: true, limits: [],
  tool: { name: "ffprobe", version: "8.1.2", sha256: "b".repeat(64), origin: "system" },
};
const PREVIEW: VideoTrimPreview = {
  version: 1, artifact_id: "sha256:clip", artifact_sha256: "a".repeat(64),
  requested_start_seconds: 1.43, requested_end_seconds: 2, keep_audio: true,
  start_seconds: 1, keyframe_seconds: 1, from_beginning: false, keeps_whole_video: false,
  audio_streams_kept: 1, omitted_streams: 0, format: "mp4", media_type: "video/mp4",
};
const RESULT = {
  result_artifact_id: "sha256:part", in_library: true, action: "trim",
  source_artifact_id: "sha256:clip", source_sha256: "a".repeat(64),
  requested_start_seconds: 1.43, requested_end_seconds: 2, actual_start_seconds: 1, keyframe_seconds: 1,
  actual_end_seconds: 2.04, from_beginning: false, keep_audio: true, format: "mp4", media_type: "video/mp4",
  video: {
    codec: "h264", start_seconds: 0.023, first_shown_seconds: 0.023, end_seconds: 1.063, frames_stored: 11,
    display_width: 64, display_height: 48, rotation: 0, sample_aspect: null,
  },
  audio: [{ codec: "aac", start_seconds: 0, packets: 49 }],
  omitted_streams: 0, tool: { name: "ffmpeg" }, measured_with: { name: "ffprobe" },
} satisfies VideoTrimResult;
const STAMP = "2026-10-06T00:00:00Z";
const STALE = "Where this cut starts has changed. Check the cut again.";

function job(status: Job["status"], fields: Partial<Job> = {}): Job {
  return {
    id: "job-trim", kind: "media_utility", status, run_id: null, progress: 0, phase: status,
    payload_json: {}, result_json: {}, error: null, attempt: 1, cancellable: true, created_at: STAMP,
    updated_at: STAMP, started_at: null, completed_at: null, ...fields,
  } as Job;
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.videoProbe).mockResolvedValue(PROBE);
  vi.mocked(api.videoTrimPreview).mockResolvedValue(PREVIEW);
  vi.mocked(api.trimVideo).mockResolvedValue(job("queued"));
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("queued"));
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  localStorage.clear();
});

async function open(probe: VideoProbe = PROBE) {
  vi.mocked(api.videoProbe).mockResolvedValue(probe);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}>
    <VideoTrimButton artifactId="sha256:clip" source="/api/artifacts/sha256:clip/content" />
  </QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Trim this video" }));
  await screen.findByLabelText("Start (seconds)");
}

function enter(label: string, value: string) {
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
}

async function checkCut() {
  fireEvent.click(screen.getByRole("button", { name: "Check the cut" }));
  await screen.findByText(/Its real end is measured once it is made\./);
}

/** Check the cut from 1.43 s to the end, then trim it. */
async function trimChecked() {
  await open();
  enter("Start (seconds)", "1.43");
  await checkCut();
  fireEvent.click(screen.getByRole("button", { name: "Trim" }));
  await waitFor(() => expect(api.trimVideo).toHaveBeenCalled());
}

it("trims only a checked cut, and says where the copy really starts", async () => {
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("running", { phase: "Copying the chosen part" }));
  await open();
  expect(screen.getByText(/A copy can only begin on a keyframe, a frame the video stores whole/)).toBeInTheDocument();
  enter("Start (seconds)", "1.43");

  const trim = screen.getByRole("button", { name: "Trim" });
  expect(trim).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(trim);
  expect(api.trimVideo).not.toHaveBeenCalled();

  await checkCut();
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 1.43, 2, true);
  expect(screen.getByText("It starts at the keyframe at 1.000 s, 0.430 s before your start.")).toBeInTheDocument();
  expect(screen.getByText("It ends near 2.000 s. Its real end is measured once it is made.")).toBeInTheDocument();
  expect(screen.getByText("Saved as a new MP4 video. The original is not changed.")).toBeInTheDocument();
  expect(screen.getByText("It keeps its sound.")).toBeInTheDocument();
  expect(screen.getByRole("dialog").textContent).not.toMatch(/exact/i);
  expect(trim).toHaveAttribute("aria-disabled", "false");

  fireEvent.click(trim);
  await waitFor(() => expect(api.trimVideo).toHaveBeenCalledWith("sha256:clip", {
    start_seconds: 1.43, end_seconds: 2, keep_audio: true, shown_start_seconds: 1,
  }));
  expect(await screen.findByText("Copying the chosen part…")).toBeInTheDocument();
  expect(screen.getByRole("dialog").textContent).not.toMatch(/exact/i);
});

it("takes the preview away once the cut changes, and keeps both ends within the video", async () => {
  await open();
  enter("Start (seconds)", "1.43");
  await checkCut();
  const trim = screen.getByRole("button", { name: "Trim" });
  expect(trim).toHaveAttribute("aria-disabled", "false");

  enter("End (seconds)", "1.9");
  expect(screen.queryByText(/0\.430 s before your start/)).not.toBeInTheDocument();
  expect(trim).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(trim);
  expect(api.trimVideo).not.toHaveBeenCalled();

  enter("End (seconds)", "2");
  expect(screen.getByText("It starts at the keyframe at 1.000 s, 0.430 s before your start.")).toBeInTheDocument();
  expect(api.videoTrimPreview).toHaveBeenCalledTimes(1);

  enter("End (seconds)", "25");
  fireEvent.blur(screen.getByLabelText("End (seconds)"));
  expect(screen.getByLabelText("End (seconds)")).toHaveValue(2);
});

it("leaves the sound out when it cannot be copied, and offers no choice about it", async () => {
  vi.mocked(api.videoTrimPreview).mockResolvedValue({
    ...PREVIEW, requested_start_seconds: 0.5, keep_audio: false, audio_streams_kept: 0,
  });
  await open({ ...PROBE, can_keep_audio: false, limits: ["audio-codec-unsupported"] });

  expect(screen.queryByRole("checkbox", { name: "Keep the sound" })).not.toBeInTheDocument();
  expect(screen.getByText("Its sound cannot be copied unchanged, so the new video has none.")).toBeInTheDocument();
  enter("Start (seconds)", "0.5");
  await checkCut();
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 0.5, 2, false);
  expect(screen.getByText("Without sound.")).toBeInTheDocument();
});

it("offers to keep the sound only when the video has some", async () => {
  vi.mocked(api.videoTrimPreview).mockResolvedValue({
    ...PREVIEW, requested_start_seconds: 0, start_seconds: 0, keyframe_seconds: 0, keep_audio: false,
    audio_streams_kept: 0,
  });
  await open();
  const sound = screen.getByRole("checkbox", { name: "Keep the sound" });
  expect(sound).toBeChecked();
  fireEvent.click(sound);
  await checkCut();
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 0, 2, false);
  cleanup();

  await open({ ...PROBE, audio: [] });
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  expect(screen.queryByText(/sound cannot be copied/)).not.toBeInTheDocument();
});

it("says when a cut has to start at the beginning, and refuses one that keeps the whole video", async () => {
  vi.mocked(api.videoTrimPreview)
    .mockResolvedValueOnce({
      // The first picture comes a little after the beginning, at its first keyframe.
      ...PREVIEW, requested_start_seconds: 0.25, start_seconds: 0, keyframe_seconds: 0.04, from_beginning: true,
      omitted_streams: 2, format: "matroska", media_type: "video/x-matroska",
    })
    .mockResolvedValueOnce({
      ...PREVIEW, requested_start_seconds: 0, start_seconds: 0, keyframe_seconds: 0, from_beginning: true,
      keeps_whole_video: true,
    });
  await open();
  enter("Start (seconds)", "0.25");
  await checkCut();
  expect(screen.getByText(
    "It starts at the beginning of the video, 0.250 s before your start, because your start comes before its second keyframe.",
  )).toBeInTheDocument();
  expect(screen.getByText("Saved as a new Matroska video. The original is not changed.")).toBeInTheDocument();
  expect(screen.getByText("Subtitles, cover pictures and other streams are left out.")).toBeInTheDocument();
  expect(screen.getByText("Some browsers cannot play Matroska videos; it can still be downloaded.")).toBeInTheDocument();

  enter("Start (seconds)", "0");
  await checkCut();
  expect(screen.getByText("It starts at the beginning.")).toBeInTheDocument();
  expect(screen.getByText(
    "That would keep the whole video as it is. Choose an earlier end, or leave out the sound.",
  )).toBeInTheDocument();
  const trim = screen.getByRole("button", { name: "Trim" });
  expect(trim).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(trim);
  expect(api.trimVideo).not.toHaveBeenCalled();
});

it("suggests leaving out the sound only when a cut that keeps the whole video keeps some", async () => {
  vi.mocked(api.videoTrimPreview).mockResolvedValue({
    ...PREVIEW, requested_start_seconds: 0, start_seconds: 0, keyframe_seconds: 0, from_beginning: true,
    keeps_whole_video: true, keep_audio: false, audio_streams_kept: 0,
  });
  await open({ ...PROBE, audio: [] });
  await checkCut();

  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 0, 2, false);
  expect(screen.getByText("That would keep the whole video as it is. Choose an earlier end.")).toBeInTheDocument();
  expect(screen.queryByText(/leave out the sound/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Trim" })).toHaveAttribute("aria-disabled", "true");
});

it("says why a cut was refused until the cut changes, and offers no trim for it", async () => {
  const beforePicture = "That part ends before the video's first picture.";
  vi.mocked(api.videoTrimPreview).mockRejectedValue(
    new ApiError(422, beforePicture, beforePicture, "video-trim-range-before-picture"),
  );
  await open();
  enter("End (seconds)", "0.2");
  fireEvent.click(screen.getByRole("button", { name: "Check the cut" }));

  expect(await screen.findByText(beforePicture)).toHaveAttribute("role", "alert");
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 0, 0.2, true);
  const trim = screen.getByRole("button", { name: "Trim" });
  expect(trim).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(trim);
  expect(api.trimVideo).not.toHaveBeenCalled();

  enter("End (seconds)", "1");
  expect(screen.queryByText(beforePicture)).not.toBeInTheDocument();
});

it("names each stage while it trims, and stops when asked", async () => {
  await trimChecked();
  expect(await screen.findByText("Waiting its turn…")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Trim" })).toHaveAttribute("aria-disabled", "true");

  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("running", { phase: "Finding the keyframe" }));
  expect(await screen.findByText("Finding the keyframe…", {}, { timeout: 3000 })).toBeInTheDocument();
  // A stage the dialog does not name is still shown as work going on.
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("running", { phase: "Running" }));
  expect(await screen.findByText("Trimming…", {}, { timeout: 3000 })).toBeInTheDocument();

  vi.mocked(api.cancelJob).mockResolvedValue(job("running", { phase: "Running" }));
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("cancelled"));
  fireEvent.click(screen.getByRole("button", { name: "Stop trimming" }));
  expect(await screen.findByText("Trimming was stopped. Nothing was saved.")).toBeInTheDocument();
  expect(api.cancelJob).toHaveBeenCalledWith("job-trim");
  expect(screen.getByRole("button", { name: "Stop trimming" })).toHaveAttribute("aria-disabled", "true");
  expect(screen.queryByText("Trimming…")).not.toBeInTheDocument();
});

it("keeps the stop button, and the focus on it, once the trim has ended", async () => {
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("running", { phase: "Copying the chosen part" }));
  await trimChecked();
  const stop = await screen.findByRole("button", { name: "Stop trimming" });
  expect(stop).toHaveAttribute("aria-disabled", "false");
  stop.focus();
  expect(document.activeElement).toBe(stop);

  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("complete", { phase: "Video trimmed", result_json: RESULT }));
  expect(await screen.findByText(
    "Saved a new video from 1.000 s to 2.040 s of the original (you chose 1.430 s to 2.000 s). It is in the Media Library.",
    {}, { timeout: 3000 },
  )).toBeInTheDocument();
  expect(screen.queryByText("Copying the chosen part…")).not.toBeInTheDocument();
  expect(stop).toBeInTheDocument();
  expect(stop).toHaveAttribute("aria-disabled", "true");
  expect(document.activeElement).toBe(stop);
  fireEvent.click(stop);
  expect(api.cancelJob).not.toHaveBeenCalled();
});

it("says a trim already being saved will finish when it can no longer be stopped", async () => {
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("running"));
  vi.mocked(api.cancelJob).mockRejectedValue(new ApiError(
    409, "That job can no longer be cancelled.", "That job can no longer be cancelled.", "job-not-cancellable",
  ));
  await trimChecked();

  fireEvent.click(await screen.findByRole("button", { name: "Stop trimming" }));
  expect(await screen.findByText("It was already being saved, so it will finish.")).toBeInTheDocument();
  expect(api.cancelJob).toHaveBeenCalledWith("job-trim");
});

it("reports the part that was saved beside the part that was chosen", async () => {
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("complete", { phase: "Video trimmed", result_json: RESULT }));
  await trimChecked();

  expect(await screen.findByText(
    "Saved a new video from 1.000 s to 2.040 s of the original (you chose 1.430 s to 2.000 s). It is in the Media Library.",
  )).toBeInTheDocument();
  expect(screen.getByText("Its picture begins 0.023 s after its sound.")).toBeInTheDocument();
  expect(screen.queryByText(/Recently Deleted/)).not.toBeInTheDocument();
});

it("points to Recently Deleted when the same video is already there", async () => {
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("complete", {
    phase: "Video trimmed",
    result_json: { ...RESULT, in_library: false, from_beginning: true, actual_start_seconds: 0, keep_audio: false },
  }));
  await trimChecked();

  expect(await screen.findByText(
    "Saved a new video from the beginning to 2.040 s of the original (you chose 1.430 s to 2.000 s).",
  )).toBeInTheDocument();
  expect(screen.getByText(
    "The same video is already in Recently Deleted. Restore it there to see it in the Media Library.",
  )).toBeInTheDocument();
  expect(screen.queryByText(/picture begins/)).not.toBeInTheDocument();
});

it("follows a second trim in the same dialog, and shows only what that one made", async () => {
  const firstSaved = "Saved a new video from 1.000 s to 2.040 s of the original (you chose 1.430 s to 2.000 s). It is in the Media Library.";
  const first = job("complete", { phase: "Video trimmed", result_json: RESULT });
  let second = job("running", { id: "job-second", phase: "Copying the chosen part" });
  vi.mocked(api.videoUtilityJob).mockImplementation(async (id) => (id === "job-second" ? second : first));
  await trimChecked();
  expect(await screen.findByText(firstSaved)).toBeInTheDocument();

  let queue: (queued: Job) => void = () => undefined;
  vi.mocked(api.trimVideo).mockReturnValueOnce(new Promise<Job>((resolve) => { queue = resolve; }));
  vi.mocked(api.videoTrimPreview).mockResolvedValue({ ...PREVIEW, requested_end_seconds: 1.5 });
  enter("End (seconds)", "1.5");
  await checkCut();
  fireEvent.click(screen.getByRole("button", { name: "Trim" }));

  // While the second trim is being sent, the first one's outcome is not shown as if it were this one's.
  expect(await screen.findByText("Trimming…")).toBeInTheDocument();
  expect(api.trimVideo).toHaveBeenLastCalledWith("sha256:clip", {
    start_seconds: 1.43, end_seconds: 1.5, keep_audio: true, shown_start_seconds: 1,
  });
  expect(screen.queryByText(firstSaved)).not.toBeInTheDocument();

  queue(job("queued", { id: "job-second" }));
  expect(await screen.findByText("Copying the chosen part…")).toBeInTheDocument();
  expect(screen.queryByText(firstSaved)).not.toBeInTheDocument();

  second = job("complete", {
    id: "job-second", phase: "Video trimmed",
    result_json: { ...RESULT, requested_end_seconds: 1.5, actual_end_seconds: 1.54 },
  });
  expect(await screen.findByText(
    "Saved a new video from 1.000 s to 1.540 s of the original (you chose 1.430 s to 1.500 s). It is in the Media Library.",
    {}, { timeout: 3000 },
  )).toBeInTheDocument();
  expect(screen.queryByText(firstSaved)).not.toBeInTheDocument();
  expect(api.videoUtilityJob).toHaveBeenCalledWith("job-second", expect.anything());
});

it.each([
  {
    status: "failed",
    ended: job("failed", { phase: "Video not trimmed", error: "The chosen part could not be copied." }),
    text: "The chosen part could not be copied.",
  },
  {
    status: "interrupted",
    ended: job("interrupted"),
    text: "Trimming was interrupted when the app stopped. It starts again when the app restarts.",
  },
])("says what became of a trim that ended $status", async ({ ended, text }) => {
  vi.mocked(api.videoUtilityJob).mockResolvedValue(ended);
  await trimChecked();
  expect(await screen.findByText(text)).toBeInTheDocument();
});

it("takes the preview away when where the cut starts has changed since it was checked", async () => {
  vi.mocked(api.trimVideo).mockRejectedValue(new ApiError(409, STALE, STALE, "video-trim-preview-stale"));
  await trimChecked();

  expect(await screen.findByText(STALE)).toBeInTheDocument();
  expect(screen.queryByText(/0\.430 s before your start/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Trim" })).toHaveAttribute("aria-disabled", "true");
  expect(api.videoUtilityJob).not.toHaveBeenCalled();
});

it("names why a video cannot be trimmed instead of offering it", async () => {
  vi.mocked(api.videoProbe).mockResolvedValue({
    ...PROBE, can_trim: false, limits: ["video-duration-too-long", "video-rotation-unsupported"],
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}>
    <VideoTrimButton artifactId="sha256:clip" source="/api/artifacts/sha256:clip/content" />
  </QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Trim this video" }));

  expect(await screen.findByText("This video cannot be trimmed.")).toBeInTheDocument();
  expect(screen.getByText("It is longer than the video utilities read.")).toBeInTheDocument();
  expect(screen.getByText("It is turned by an angle other than a quarter turn.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Trim" })).not.toBeInTheDocument();
});

it("takes a time from the player only while the player is on screen", async () => {
  localStorage.setItem(SENSITIVE_MEDIA_KEY, "hide");
  await open();

  expect(screen.getByText("This video is hidden.")).toBeInTheDocument();
  expect(screen.queryByLabelText("Video to trim")).not.toBeInTheDocument();
  const here = screen.getByRole("button", { name: "Start here" });
  expect(here).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByRole("button", { name: "End here" })).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(here);
  expect(screen.getByLabelText("Start (seconds)")).toHaveValue(0);

  fireEvent.click(screen.getByRole("button", { name: "Show video" }));
  const video = screen.getByLabelText("Video to trim");
  Object.defineProperty(video, "currentTime", { configurable: true, value: 0.4567 });
  expect(here).toHaveAttribute("aria-disabled", "false");
  fireEvent.click(here);
  expect(screen.getByLabelText("Start (seconds)")).toHaveValue(0.457);
});

it("takes no time from a blurred player, which cannot be played until it is shown", async () => {
  localStorage.setItem(SENSITIVE_MEDIA_KEY, "blur");
  await open();

  expect(screen.getByText("This video is blurred.")).toBeInTheDocument();
  const covered = screen.getByLabelText("Video to trim");
  Object.defineProperty(covered, "currentTime", { configurable: true, value: 0.4567 });
  const start = screen.getByRole("button", { name: "Start here" });
  const end = screen.getByRole("button", { name: "End here" });
  expect(start).toHaveAttribute("aria-disabled", "true");
  expect(end).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(start);
  fireEvent.click(end);
  expect(screen.getByLabelText("Start (seconds)")).toHaveValue(0);
  expect(screen.getByLabelText("End (seconds)")).toHaveValue(2);

  fireEvent.click(screen.getByRole("button", { name: "Show video" }));
  const video = screen.getByLabelText("Video to trim");
  expect(video.closest("[inert]")).toBeNull();
  Object.defineProperty(video, "currentTime", { configurable: true, value: 1.5 });
  expect(start).toHaveAttribute("aria-disabled", "false");
  expect(end).toHaveAttribute("aria-disabled", "false");
  fireEvent.click(end);
  expect(screen.getByLabelText("End (seconds)")).toHaveValue(1.5);
});
