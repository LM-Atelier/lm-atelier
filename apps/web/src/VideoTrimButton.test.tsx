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
    pixel_format: "yuv420p", color_range: null, color_space: null, color_transfer: null, color_primaries: null,
    field_order: "progressive",
  },
  audio: [{ index: 1, codec: "aac", channels: 2, sample_rate: 48000 }],
  omitted_streams: 0, can_save_frame: true, can_trim: true, can_keep_audio: true, limits: [],
  can_trim_exact: true, can_keep_audio_exact: true, exact_trim_limits: [],
  tool: { name: "ffprobe", version: "8.1.2", sha256: "b".repeat(64), origin: "system" },
};
const PREVIEW: VideoTrimPreview = {
  version: 1, mode: "copy", artifact_id: "sha256:clip", artifact_sha256: "a".repeat(64),
  requested_start_seconds: 1.43, requested_end_seconds: 2, keep_audio: true,
  start_seconds: 1, keyframe_seconds: 1, from_beginning: false, keeps_whole_video: false,
  last_frame_seconds: null, end_seconds: null, frame_count: null, began_at_first_picture: false,
  audio_streams_kept: 1, omitted_streams: 0, format: "mp4", media_type: "video/mp4",
};
/** The same part checked as an exact cut: the frame shown at 1.43 s began at 1.4 s. */
const EXACT: VideoTrimPreview = {
  ...PREVIEW, mode: "exact", start_seconds: 1.4, last_frame_seconds: 1.9, end_seconds: 2, frame_count: 6,
};
const RESULT = {
  result_artifact_id: "sha256:part", in_library: true, action: "trim", mode: "copy",
  source_artifact_id: "sha256:clip", source_sha256: "a".repeat(64),
  requested_start_seconds: 1.43, requested_end_seconds: 2, actual_start_seconds: 1, keyframe_seconds: 1,
  actual_end_seconds: 2.04, from_beginning: false, keep_audio: true, format: "mp4", media_type: "video/mp4",
  video: {
    codec: "h264", start_seconds: 0.023, first_shown_seconds: 0.023, end_seconds: 1.063, frames_stored: 11,
    display_width: 64, display_height: 48, rotation: 0, sample_aspect: null,
  },
  audio: [{ codec: "aac", start_seconds: 0, packets: 49 }],
  omitted_streams: 0, frames: null, quality: null, encoding: null,
  tool: { name: "ffmpeg" }, measured_with: { name: "ffprobe" },
} satisfies VideoTrimResult;
const EXACT_RESULT = {
  ...RESULT, mode: "exact", actual_start_seconds: 1.4, actual_end_seconds: 2,
  frames: {
    first_seconds: 1.4, last_seconds: 1.9, count: 6, began_at_first_picture: false,
    source_frames_sha256: "c".repeat(64), decoded_from_seconds: 0, checked_from_seconds: 1,
  },
  quality: { ssim_mean: 0.99873, psnr_mean_db: 47.97, psnr_lowest_db: 46.79 },
  encoding: {
    video_encoder: "libx264", preset: "medium", crf: 18, pixel_format: "yuv420p", audio_encoder: "aac",
    audio_bitrates: [192000],
  },
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
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 1.43, 2, true, "copy");
  expect(screen.getByText("It starts at the keyframe at 1.000 s, 0.430 s before your start.")).toBeInTheDocument();
  expect(screen.getByText("It ends near 2.000 s. Its real end is measured once it is made.")).toBeInTheDocument();
  expect(screen.getByText("Saved as a new MP4 video. The original is not changed.")).toBeInTheDocument();
  expect(screen.getByText("It keeps its sound.")).toBeInTheDocument();
  // A copy that starts early points to the cut that would not, and claims nothing exact of itself.
  expect(screen.getByText(
    "To start on the frame at your start instead, choose Cut on the exact frames and check again.",
  )).toBeInTheDocument();
  expect(screen.getByRole("radio", { name: "Keep the original quality" })).toBeChecked();
  expect(trim).toHaveAttribute("aria-disabled", "false");

  fireEvent.click(trim);
  await waitFor(() => expect(api.trimVideo).toHaveBeenCalledWith("sha256:clip", {
    start_seconds: 1.43, end_seconds: 2, keep_audio: true, shown_start_seconds: 1, mode: "copy",
  }));
  expect(await screen.findByText("Copying the chosen part…")).toBeInTheDocument();
});

it("cuts on the exact frames when chosen, bound to the frames the check showed", async () => {
  vi.mocked(api.videoTrimPreview).mockResolvedValue(EXACT);
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("running", { phase: "Re-encoding the chosen part" }));
  await open();
  enter("Start (seconds)", "1.43");
  fireEvent.click(screen.getByRole("radio", { name: "Cut on the exact frames" }));
  expect(screen.getByRole("radio", { name: "Cut on the exact frames" })).toBeChecked();
  fireEvent.click(screen.getByRole("button", { name: "Check the cut" }));

  expect(await screen.findByText("It starts on the frame shown at your start, which begins at 1.400 s.")).toBeInTheDocument();
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 1.43, 2, true, "exact");
  expect(screen.getByText(
    "It ends after the frame at 1.900 s, the last one shown before your end: 6 frames, 0.600 s in all.",
  )).toBeInTheDocument();
  expect(screen.getByText(/close to the original but not identical, and it takes longer than a copy/)).toBeInTheDocument();
  expect(screen.getByText("Its sound is re-encoded as AAC and cut with the picture.")).toBeInTheDocument();
  expect(screen.queryByText(/Its real end is measured/)).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Trim" }));
  await waitFor(() => expect(api.trimVideo).toHaveBeenCalledWith("sha256:clip", {
    start_seconds: 1.43, end_seconds: 2, keep_audio: true, shown_start_seconds: 1.4, mode: "exact",
    shown_frame_count: 6,
  }));
  expect(await screen.findByText("Re-encoding the chosen part…")).toBeInTheDocument();
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("running", { phase: "Comparing it with the original" }));
  expect(await screen.findByText("Comparing it with the original…", {}, { timeout: 3000 })).toBeInTheDocument();
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("complete", { phase: "Video trimmed", result_json: EXACT_RESULT }));
  expect(await screen.findByText(
    "Saved a new video of 6 frames, from 1.400 s to 2.000 s of the original (you chose 1.430 s to 2.000 s). It is in the Media Library.",
    {}, { timeout: 3000 },
  )).toBeInTheDocument();
  expect(screen.getByText(
    "Measured against the original frame by frame: SSIM 0.999 on average (1 is identical), and no frame below 46.8 dB PSNR.",
  )).toBeInTheDocument();
});

it("says when an exact cut starts on the frame asked for, or on the first picture", async () => {
  vi.mocked(api.videoTrimPreview)
    .mockResolvedValueOnce({ ...EXACT, requested_start_seconds: 1.4 })
    .mockResolvedValueOnce({
      ...EXACT, requested_start_seconds: 0, start_seconds: 0.5, began_at_first_picture: true, frame_count: 1,
      last_frame_seconds: 0.5, end_seconds: 0.6, keeps_whole_video: true,
    });
  await open();
  fireEvent.click(screen.getByRole("radio", { name: "Cut on the exact frames" }));
  enter("Start (seconds)", "1.4");
  fireEvent.click(screen.getByRole("button", { name: "Check the cut" }));
  expect(await screen.findByText("It starts on the frame at 1.400 s, your start.")).toBeInTheDocument();

  enter("Start (seconds)", "0");
  fireEvent.click(screen.getByRole("button", { name: "Check the cut" }));
  expect(await screen.findByText(
    "Your start comes before the video's first picture, so it starts on that picture, at 0.500 s. The sound before it is left out.",
  )).toBeInTheDocument();
  expect(screen.getByText(/the last one shown before your end: 1 frame, 0\.100 s in all\./)).toBeInTheDocument();
  expect(screen.getByText(
    "That keeps every frame of the video. Choose a later start or an earlier end. To leave out only its sound, keep the original quality instead.",
  )).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Trim" })).toHaveAttribute("aria-disabled", "true");
});

it("tells a silent video only to choose another part when an exact cut keeps every frame", async () => {
  vi.mocked(api.videoTrimPreview).mockResolvedValue({
    ...EXACT, requested_start_seconds: 0, start_seconds: 0, keep_audio: false, audio_streams_kept: 0,
    keeps_whole_video: true,
  });
  await open({ ...PROBE, audio: [] });
  fireEvent.click(screen.getByRole("radio", { name: "Cut on the exact frames" }));
  fireEvent.click(screen.getByRole("button", { name: "Check the cut" }));

  expect(await screen.findByText(
    "That keeps every frame of the video. Choose a later start or an earlier end.",
  )).toBeInTheDocument();
  expect(screen.queryByText(/leave out only its sound/)).not.toBeInTheDocument();
});

it("takes the preview away when the way to cut changes", async () => {
  vi.mocked(api.videoTrimPreview).mockResolvedValueOnce(PREVIEW).mockResolvedValueOnce(EXACT);
  await open();
  enter("Start (seconds)", "1.43");
  await checkCut();

  fireEvent.click(screen.getByRole("radio", { name: "Cut on the exact frames" }));
  expect(screen.queryByText(/0\.430 s before your start/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Trim" })).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(screen.getByRole("radio", { name: "Keep the original quality" }));
  expect(screen.getByText("It starts at the keyframe at 1.000 s, 0.430 s before your start.")).toBeInTheDocument();
  expect(api.videoTrimPreview).toHaveBeenCalledTimes(1);
});

it("offers no exact cut for a video it would change, and says why", async () => {
  await open({
    ...PROBE, can_trim_exact: false, can_keep_audio_exact: false,
    exact_trim_limits: ["exact-trim-picture-unsupported", "exact-trim-turned-unsupported", "exact-trim-audio-unsupported"],
  });

  const exact = screen.getByRole("radio", { name: "Cut on the exact frames" });
  expect(exact).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(exact);
  expect(screen.getByRole("radio", { name: "Keep the original quality" })).toBeChecked();
  expect(screen.getByText("This video cannot be cut on exact frames.")).toBeInTheDocument();
  expect(screen.getByText(/such as 10-bit, HDR, full-range or interlaced video\./)).toBeInTheDocument();
  expect(screen.getByText("It is turned, and an exact cut cannot keep the turn yet.")).toBeInTheDocument();
  // Sound that only an exact cut cannot keep is no reason the cut is not offered.
  expect(screen.queryByText(/an exact cut leaves it out/)).not.toBeInTheDocument();
  enter("Start (seconds)", "1.43");
  await checkCut();
  expect(screen.queryByText(/choose Cut on the exact frames/)).not.toBeInTheDocument();
});

it("offers to keep the sound in an exact cut only when it can be re-encoded", async () => {
  vi.mocked(api.videoTrimPreview).mockResolvedValue({ ...EXACT, keep_audio: false, audio_streams_kept: 0 });
  await open({ ...PROBE, can_keep_audio_exact: false, exact_trim_limits: ["exact-trim-audio-unsupported"] });
  expect(screen.getByRole("checkbox", { name: "Keep the sound" })).toBeChecked();

  fireEvent.click(screen.getByRole("radio", { name: "Cut on the exact frames" }));
  expect(screen.queryByRole("checkbox", { name: "Keep the sound" })).not.toBeInTheDocument();
  expect(screen.getByText("Its sound cannot be re-encoded for an exact cut, so the new video has none.")).toBeInTheDocument();
  enter("Start (seconds)", "1.43");
  fireEvent.click(screen.getByRole("button", { name: "Check the cut" }));
  await screen.findByText("Without sound.");
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 1.43, 2, false, "exact");
});

it("says an exact cut decoded identical when its PSNR is infinite", async () => {
  vi.mocked(api.videoTrimPreview).mockResolvedValue(EXACT);
  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("complete", {
    phase: "Video trimmed",
    result_json: { ...EXACT_RESULT, quality: { ssim_mean: 1, psnr_mean_db: null, psnr_lowest_db: null } },
  }));
  await open();
  enter("Start (seconds)", "1.43");
  fireEvent.click(screen.getByRole("radio", { name: "Cut on the exact frames" }));
  fireEvent.click(screen.getByRole("button", { name: "Check the cut" }));
  await screen.findByText(/It starts on the frame shown at your start/);
  fireEvent.click(screen.getByRole("button", { name: "Trim" }));

  expect(await screen.findByText("Every frame decoded identical to the original's.")).toBeInTheDocument();
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
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 0.5, 2, false, "copy");
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
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 0, 2, false, "copy");
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

  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 0, 2, false, "copy");
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
  expect(api.videoTrimPreview).toHaveBeenCalledWith("sha256:clip", 0, 0.2, true, "copy");
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
    start_seconds: 1.43, end_seconds: 1.5, keep_audio: true, shown_start_seconds: 1, mode: "copy",
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

it("keeps an accepted trim when its progress cannot be read, and reads it again only when asked", async () => {
  vi.mocked(api.videoUtilityJob)
    .mockResolvedValueOnce(job("running", { phase: "Copying the chosen part" }))
    .mockRejectedValueOnce(new ApiError(503, "The app is busy.", "The app is busy.", "busy"));
  await trimChecked();

  expect(await screen.findByText(
    "Trimming was accepted, but how it is going could not be read: The app is busy.",
  )).toBeInTheDocument();
  // The reading stops rather than repeating, and the cut is not sent again.
  const reads = vi.mocked(api.videoUtilityJob).mock.calls.length;
  await new Promise((resolve) => setTimeout(resolve, 700));
  expect(api.videoUtilityJob).toHaveBeenCalledTimes(reads);
  expect(screen.queryByText("Copying the chosen part…")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Trim" })).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByRole("button", { name: "Stop trimming" })).toHaveAttribute("aria-disabled", "false");

  vi.mocked(api.videoUtilityJob).mockResolvedValue(job("complete", { phase: "Video trimmed", result_json: RESULT }));
  fireEvent.click(screen.getByRole("button", { name: "Check again" }));
  expect(await screen.findByText(
    "Saved a new video from 1.000 s to 2.040 s of the original (you chose 1.430 s to 2.000 s). It is in the Media Library.",
  )).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Check again" })).not.toBeInTheDocument();
  expect(api.videoUtilityJob).toHaveBeenCalledTimes(reads + 1);
  expect(api.trimVideo).toHaveBeenCalledTimes(1);
});

it("says the tools it runs were found on this computer, and only when they were", async () => {
  const found = "The video utilities use the ffmpeg and ffprobe found on this computer, which the app has not checked.";
  await open();
  expect(screen.getByText(found)).toBeInTheDocument();

  cleanup();
  await open({ ...PROBE, tool: { ...PROBE.tool, origin: "provided" } });
  expect(screen.queryByText(found)).not.toBeInTheDocument();
});
