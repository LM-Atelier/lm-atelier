/** The generation record control: read only when asked, shown plainly, saved byte for byte. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ArtifactPart } from "./ArtifactPart";
import { api } from "./api";
import { downloadBytes } from "./format";
import { GenerationRecordButton } from "./GenerationRecordButton";
import { recordBytes } from "./generationRecordFixtures";
import type { MessagePart } from "./types";

vi.mock("./api", () => ({
  api: { generationRecord: vi.fn(), generationRecordBundle: vi.fn(), replayResult: vi.fn(), outputRecipeDraft: vi.fn() },
}));
// The recipe dialogs have their own tests; here only what they are given matters.
vi.mock("./GenerationEditRecipeDialog", () => ({
  GenerationEditRecipeDialog: ({ runId }: { runId: string }) => (
    <div role="dialog" aria-label="Keep this edit as a recipe">{runId}</div>
  ),
}));
vi.mock("./RecipeDraftDialog", () => ({
  RecipeDraftDialog: ({ title, draft }: { title: string; draft: { data?: { name: string } } }) => (
    <div role="dialog" aria-label={title}>{draft.data?.name}</div>
  ),
}));
vi.mock("./format", async (original) => ({
  ...(await original<typeof import("./format")>()),
  downloadBytes: vi.fn(),
}));

const SHA = "a".repeat(64);
const generationRecord = vi.mocked(api.generationRecord);
const generationRecordBundle = vi.mocked(api.generationRecordBundle);
const replayResult = vi.mocked(api.replayResult);
const outputRecipeDraft = vi.mocked(api.outputRecipeDraft);
const saved = vi.mocked(downloadBytes);

function withQueries(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function openRecord() {
  fireEvent.click(screen.getByRole("button", { name: "Generation record of this image" }));
}

beforeEach(() => {
  generationRecord.mockReset();
  generationRecordBundle.mockReset();
  replayResult.mockReset();
  replayResult.mockRejectedValue(Object.assign(new Error("not found"), { code: "replay-result-not-found" }));
  outputRecipeDraft.mockReset();
  saved.mockReset();
});
afterEach(cleanup);

describe("the generation record dialog", () => {
  it("reads nothing until it is opened", () => {
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));

    expect(generationRecord).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("starts without the prompt and says what was left out and why", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));

    openRecord();

    expect(await screen.findByText("Left out, as you chose.")).toBeInTheDocument();
    expect(generationRecord).toHaveBeenCalledWith("run_1", `sha256:${SHA}`, false, expect.any(AbortSignal));
    expect(screen.getByRole("checkbox", { name: /Include the prompt/ })).not.toBeChecked();
    expect(screen.getByText('The setting "house_style"')).toBeInTheDocument();
    expect(screen.getByText("This generation did not keep a frozen copy of its inputs.")).toBeInTheDocument();
    expect(screen.getByText("Picture edit")).toBeInTheDocument();
  });

  it("reads the record again with the prompt once that is chosen", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");

    generationRecord.mockResolvedValue(recordBytes({
      prompt: { included: true, negative: null, omitted_reason: null, positive: "a red cube" },
      removed: [],
      reproducibility: { missing: ["frozen_snapshot_absent"], status: "incomplete" },
    }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Include the prompt/ }));

    expect(await screen.findByText("Included")).toBeInTheDocument();
    expect(generationRecord).toHaveBeenLastCalledWith("run_1", `sha256:${SHA}`, true, expect.any(AbortSignal));
  });

  it("saves exactly the bytes it showed, under the output's name", async () => {
    const bytes = recordBytes();
    generationRecord.mockResolvedValue(bytes);
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");

    fireEvent.click(screen.getByRole("button", { name: "Download record" }));

    expect(saved).toHaveBeenCalledTimes(1);
    expect(saved.mock.calls[0][0]).toBe(bytes);
    expect(saved.mock.calls[0][1]).toBe(`generation-record-${SHA.slice(0, 12)}.json`);
    expect(saved.mock.calls[0][2]).toBe("application/json");
  });

  it("saves the picture with the record it showed, named by that record's digest", async () => {
    const zipped = new ArrayBuffer(4);
    generationRecord.mockResolvedValue(recordBytes());
    generationRecordBundle.mockResolvedValue(zipped);
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");

    fireEvent.click(screen.getByRole("button", { name: "Download with picture" }));

    await waitFor(() => expect(saved).toHaveBeenCalledTimes(1));
    expect(generationRecordBundle).toHaveBeenCalledWith(
      "run_1", `sha256:${SHA}`, false, `sha256:${"d".repeat(64)}`, expect.any(AbortSignal),
    );
    expect(saved.mock.calls[0][0]).toBe(zipped);
    expect(saved.mock.calls[0][1]).toBe(`generation-record-${SHA.slice(0, 12)}.zip`);
    expect(saved.mock.calls[0][2]).toBe("application/zip");
  });

  it("asks for the record with the prompt when the prompt is shown", async () => {
    const included = `sha256:${"e".repeat(64)}`;
    generationRecord.mockResolvedValue(recordBytes());
    generationRecordBundle.mockResolvedValue(new ArrayBuffer(4));
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");
    generationRecord.mockResolvedValue(recordBytes({
      digest: included,
      prompt: { included: true, negative: null, omitted_reason: null, positive: "a red cube" },
      removed: [],
      reproducibility: { missing: ["frozen_snapshot_absent"], status: "incomplete" },
    }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Include the prompt/ }));
    await screen.findByText("Included");

    fireEvent.click(screen.getByRole("button", { name: "Download with picture" }));

    await waitFor(() => expect(saved).toHaveBeenCalledTimes(1));
    expect(generationRecordBundle).toHaveBeenCalledWith("run_1", `sha256:${SHA}`, true, included, expect.any(AbortSignal));
  });

  it("holds the prompt choice still while the picture is copied", async () => {
    let finish: (bytes: ArrayBuffer) => void = () => undefined;
    generationRecord.mockResolvedValue(recordBytes());
    generationRecordBundle.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");

    fireEvent.click(screen.getByRole("button", { name: "Download with picture" }));

    expect(await screen.findByText("Copying the picture…")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Include the prompt/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Download with picture" })).toHaveAttribute("aria-disabled", "true");
    finish(new ArrayBuffer(4));
    await waitFor(() => expect(saved).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("checkbox", { name: /Include the prompt/ })).toBeEnabled();
    expect(screen.queryByText("Copying the picture…")).toBeNull();
  });

  it("stops copying the picture when the dialog closes, and saves nothing", async () => {
    let finish: (bytes: ArrayBuffer) => void = () => undefined;
    generationRecord.mockResolvedValue(recordBytes());
    generationRecordBundle.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");
    fireEvent.click(screen.getByRole("button", { name: "Download with picture" }));
    await screen.findByText("Copying the picture…");
    const signal = generationRecordBundle.mock.calls[0][4] as AbortSignal;

    fireEvent.click(screen.getByRole("button", { name: "Close generation record" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    expect(signal.aborted).toBe(true);
    finish(new ArrayBuffer(4));
    await Promise.resolve();
    expect(saved).not.toHaveBeenCalled();
  });

  it("can try again after a refusal", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    generationRecordBundle
      .mockRejectedValueOnce(Object.assign(new Error("refused"), { code: "output-recipe-bundle-unreadable" }))
      .mockResolvedValueOnce(new ArrayBuffer(4));
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");

    fireEvent.click(screen.getByRole("button", { name: "Download with picture" }));
    await screen.findByRole("alert");
    const again = screen.getByRole("button", { name: "Download with picture" });
    expect(again).toHaveAttribute("aria-disabled", "false");
    fireEvent.click(again);

    await waitFor(() => expect(saved).toHaveBeenCalledTimes(1));
    expect(generationRecordBundle).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("says when a picture is too large to save with its record", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    generationRecordBundle.mockRejectedValue(Object.assign(new Error("too large"), { code: "output-recipe-bundle-too-large" }));
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");

    fireEvent.click(screen.getByRole("button", { name: "Download with picture" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("This picture is too large to save with its record.");
  });

  it("reads the record again when it changed before the picture was saved", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    generationRecordBundle.mockRejectedValue(Object.assign(new Error("changed"), { code: "output-recipe-changed" }));
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");

    fireEvent.click(screen.getByRole("button", { name: "Download with picture" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("The record changed since it was shown.");
    await waitFor(() => expect(generationRecord).toHaveBeenCalledTimes(2));
    expect(saved).not.toHaveBeenCalled();
  });

  it("says when the picture could not be copied, and saves nothing", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    generationRecordBundle.mockRejectedValue(Object.assign(new Error("refused"), { code: "output-recipe-bundle-unreadable" }));
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Left out, as you chose.");

    fireEvent.click(screen.getByRole("button", { name: "Download with picture" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("The picture could not be saved with its record.");
    expect(generationRecord).toHaveBeenCalledTimes(1);
    expect(saved).not.toHaveBeenCalled();
  });

  it("offers a video's record on its own", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="video" />));

    fireEvent.click(screen.getByRole("button", { name: "Generation record of this video" }));

    expect(await screen.findByRole("button", { name: "Download record" })).toBeInTheDocument();
    await screen.findByText("Left out, as you chose.");
    expect(screen.queryByRole("button", { name: "Download with picture" })).toBeNull();
  });

  it("says whether a run generated again from a record came out the same", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    replayResult.mockResolvedValue({ state: "different", record_digest: `sha256:${"d".repeat(64)}` });
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));

    openRecord();

    expect(await screen.findByText(/it came out differently/)).toBeInTheDocument();
    expect(replayResult).toHaveBeenCalledWith("run_1", expect.any(AbortSignal));
  });

  it("says nothing about replays for a run that was not one", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));

    openRecord();
    await screen.findByText("Left out, as you chose.");

    expect(screen.queryByText(/Generated again from a record/)).toBeNull();
  });

  it("keeps a generation's settings as a recipe, in place of the record", async () => {
    generationRecord.mockResolvedValue(recordBytes({ operation: "text_to_image" }));
    outputRecipeDraft.mockResolvedValue({ name: "Neutral workflow" } as never);
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();

    fireEvent.click(await screen.findByRole("button", { name: "Keep as a recipe" }));

    const recipe = await screen.findByRole("dialog", { name: "Keep these settings as a recipe" });
    expect(await screen.findByText("Neutral workflow")).toBeInTheDocument();
    expect(outputRecipeDraft).toHaveBeenCalledWith("run_1", expect.any(AbortSignal));
    expect(screen.getAllByRole("dialog")).toEqual([recipe]);
  });

  it("keeps an edit as a Studio recipe, since its settings belong to its own picture", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    openRecord();
    await screen.findByText("Picture edit");

    fireEvent.click(screen.getByRole("button", { name: "Keep as a recipe" }));

    const recipe = await screen.findByRole("dialog", { name: "Keep this edit as a recipe" });
    expect(recipe).toHaveTextContent("run_1");
    expect(screen.getAllByRole("dialog")).toEqual([recipe]);
    expect(outputRecipeDraft).not.toHaveBeenCalled();
  });

  it("offers nothing to save when the record cannot be made", async () => {
    generationRecord.mockRejectedValue(Object.assign(new Error("refused"), { code: "output-recipe-unsafe" }));
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));

    openRecord();

    expect(await screen.findByRole("alert")).toHaveTextContent("The record could not be made for this output.");
    const download = screen.getByRole("button", { name: "Download record" });
    const withPicture = screen.getByRole("button", { name: "Download with picture" });
    expect(download).toHaveAttribute("aria-disabled", "true");
    expect(withPicture).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(download);
    fireEvent.click(withPicture);
    expect(saved).not.toHaveBeenCalled();
    expect(generationRecordBundle).not.toHaveBeenCalled();
  });

  it("returns focus to the control when closed", async () => {
    generationRecord.mockResolvedValue(recordBytes());
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));
    const control = screen.getByRole("button", { name: "Generation record of this image" });
    control.focus();
    openRecord();
    await screen.findByText("Left out, as you chose.");

    fireEvent.click(screen.getByRole("button", { name: "Close generation record" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(control).toHaveFocus();
  });
});

describe("where the control appears", () => {
  function part(type: "image" | "video", metadata: Record<string, unknown> = {}): MessagePart {
    return { id: "part-1", position: 0, type, text: null, artifact_id: `sha256:${SHA}`, metadata_json: metadata };
  }

  it("sits beside a generated picture or video that names its run", () => {
    const { rerender } = render(withQueries(
      <ArtifactPart part={part("image")} origin="generated" generationRunId="run_1" />,
    ));
    expect(screen.getByRole("button", { name: "Generation record of this image" })).toBeInTheDocument();

    rerender(withQueries(<ArtifactPart part={part("video")} origin="generated" generationRunId="run_1" />));
    expect(screen.getByRole("button", { name: "Generation record of this video" })).toBeInTheDocument();
  });

  it("is absent for an upload, a preview, an input reference or a picture with no run", () => {
    const { rerender } = render(withQueries(
      <ArtifactPart part={part("image")} origin="uploaded" generationRunId="run_1" />,
    ));
    expect(screen.queryByRole("button", { name: /Generation record/ })).toBeNull();

    rerender(withQueries(<ArtifactPart part={part("image", { preview: true })} origin="generated" generationRunId="run_1" />));
    expect(screen.queryByRole("button", { name: /Generation record/ })).toBeNull();

    rerender(withQueries(<ArtifactPart part={part("image", { input_reference: true })} origin="generated" generationRunId="run_1" />));
    expect(screen.queryByRole("button", { name: /Generation record/ })).toBeNull();

    rerender(withQueries(<ArtifactPart part={part("image")} origin="generated" />));
    expect(screen.queryByRole("button", { name: /Generation record/ })).toBeNull();
  });
});
