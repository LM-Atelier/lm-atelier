/** Checking a record file against this installation, from the model library. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { GenerationRecordCheck } from "./GenerationRecordCheck";
import { readGenerationRecordCheck } from "./generationRecord";

vi.mock("./api", () => ({ api: { checkGenerationRecord: vi.fn() } }));

const checkRecord = vi.mocked(api.checkGenerationRecord);

function answer(overrides: Record<string, unknown> = {}) {
  return {
    digest: `sha256:${"d".repeat(64)}`,
    operation: "text_to_image",
    requirements: [
      { kind: "workflow", sha256: "1".repeat(64), role: null, state: "present" },
      { kind: "model_file", sha256: "2".repeat(64), role: null, state: "inactive" },
      { kind: "lora", sha256: "3".repeat(64), role: null, state: "missing" },
    ],
    all_present: false,
    reproducibility: { missing: ["frozen_snapshot_absent"], status: "incomplete" },
    not_recorded: ["executed_graph_sha256", "runtime_version"],
    ...overrides,
  };
}

function renderCheck() {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><GenerationRecordCheck /></QueryClientProvider>);
}

function choose(content: string) {
  const file = new File([content], "generation-record.json", { type: "application/json" });
  fireEvent.change(screen.getByLabelText("Generation record file"), { target: { files: [file] } });
}

beforeEach(() => {
  checkRecord.mockReset();
});
afterEach(cleanup);

describe("checking a record", () => {
  it("sends the file's own bytes and shows each requirement's state", async () => {
    checkRecord.mockResolvedValue(answer());
    renderCheck();

    choose('{"schema":"lm-atelier-output-recipe-v1"}');

    expect(await screen.findByText("Here and ready")).toBeInTheDocument();
    expect(screen.getByText("Here, not ready")).toBeInTheDocument();
    expect(screen.getByText("Not here")).toBeInTheDocument();
    expect(screen.getByText(/Nothing was installed or changed/)).toBeInTheDocument();
    expect(screen.getByText("This generation did not keep a frozen copy of its inputs.")).toBeInTheDocument();
    const sent = checkRecord.mock.calls[0][0];
    expect(new TextDecoder().decode(sent)).toBe('{"schema":"lm-atelier-output-recipe-v1"}');
  });

  it("says plainly when everything is here", async () => {
    checkRecord.mockResolvedValue(answer({
      requirements: [{ kind: "workflow", sha256: "1".repeat(64), role: null, state: "present" }],
      all_present: true,
      reproducibility: { missing: [], status: "recorded" },
    }));
    renderCheck();

    choose("{}");

    expect(await screen.findByText(/Everything this record names is here and ready/)).toBeInTheDocument();
  });

  it("shows a fixed sentence for a file that is not a record, never the server's words", async () => {
    checkRecord.mockRejectedValue(Object.assign(new Error("server words"), { code: "output-recipe-unreadable" }));
    renderCheck();

    choose("not a record");

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This file is not a generation record this version can read.",
    );
    expect(screen.queryByText("server words")).toBeNull();
  });

  it("refuses an answer it does not recognise rather than showing part of it", async () => {
    checkRecord.mockResolvedValue(answer({
      requirements: [{ kind: "workflow", sha256: "not-a-hash", role: null, state: "present" }],
    }));
    renderCheck();

    choose("{}");

    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(screen.queryByText("Here and ready")).toBeNull();
  });

  it("closes and is ready for another file", async () => {
    checkRecord.mockResolvedValue(answer());
    renderCheck();
    choose("{}");
    await screen.findByText("Here and ready");

    fireEvent.click(screen.getByRole("button", { name: "Close the record check" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    choose("{}");
    expect(await screen.findByText("Here and ready")).toBeInTheDocument();
    expect(checkRecord).toHaveBeenCalledTimes(2);
  });
});

describe("reading the answer", () => {
  it("does not call a check all present unless every requirement is", () => {
    const read = readGenerationRecordCheck(answer({ all_present: true }));

    expect(read.allPresent).toBe(false);
    expect(read.requirements.map((item) => item.state)).toEqual(["present", "inactive", "missing"]);
  });
});
