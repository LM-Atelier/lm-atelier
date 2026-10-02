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

vi.mock("./api", () => ({ api: { generationRecord: vi.fn() } }));
vi.mock("./format", async (original) => ({
  ...(await original<typeof import("./format")>()),
  downloadBytes: vi.fn(),
}));

const SHA = "a".repeat(64);
const generationRecord = vi.mocked(api.generationRecord);
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

  it("offers nothing to save when the record cannot be made", async () => {
    generationRecord.mockRejectedValue(Object.assign(new Error("refused"), { code: "output-recipe-unsafe" }));
    render(withQueries(<GenerationRecordButton runId="run_1" artifactId={`sha256:${SHA}`} kind="image" />));

    openRecord();

    expect(await screen.findByRole("alert")).toHaveTextContent("The record could not be made for this output.");
    const download = screen.getByRole("button", { name: "Download record" });
    expect(download).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(download);
    expect(saved).not.toHaveBeenCalled();
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
