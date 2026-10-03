/** Generating a checked record again: offered only when it can be exact, and never half-started. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { discardBlankChat } from "./discardBlankChat";
import { readReplayPlan, type ReplayPlan } from "./generationRecord";
import { GenerationRecordReplay } from "./GenerationRecordReplay";

vi.mock("./api", () => ({
  api: { createChat: vi.fn(), replayGenerationRecord: vi.fn() },
}));
vi.mock("./discardBlankChat", () => ({ discardBlankChat: vi.fn() }));

const createChat = vi.mocked(api.createChat);
const discard = vi.mocked(discardBlankChat);
const replay = vi.mocked(api.replayGenerationRecord);
const CONTENT = new Uint8Array([123, 125]).buffer;

function plan(overrides: Partial<ReplayPlan> = {}): ReplayPlan {
  return {
    digest: `sha256:${"d".repeat(64)}`,
    operation: "text_to_image",
    ready: true,
    refusals: [],
    ...overrides,
  };
}

function renderReplay(value: ReplayPlan, onStarted = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <GenerationRecordReplay plan={value} content={CONTENT} onStarted={onStarted} />
    </QueryClientProvider>,
  );
  return onStarted;
}

beforeEach(() => {
  createChat.mockReset();
  discard.mockReset();
  replay.mockReset();
  createChat.mockResolvedValue({ id: "chat_again" } as Awaited<ReturnType<typeof api.createChat>>);
  discard.mockResolvedValue(undefined);
});
afterEach(cleanup);

describe("generating a record again", () => {
  it("starts the record in a new chat and opens it", async () => {
    replay.mockResolvedValue({});
    const onStarted = renderReplay(plan());

    fireEvent.click(screen.getByRole("button", { name: "Generate again" }));

    await waitFor(() => expect(onStarted).toHaveBeenCalledWith("chat_again"));
    expect(createChat).toHaveBeenCalledWith(null);
    expect(replay).toHaveBeenCalledWith("chat_again", CONTENT);
    expect(discard).not.toHaveBeenCalled();
  });

  it("says why it cannot, in fixed words, and offers nothing to start", () => {
    renderReplay(plan({
      ready: false,
      refusals: [
        { code: "replay-model-missing", sha256: null, reasons: [] },
        { code: "replay-record-unsupported", sha256: null, reasons: ["repeated_inputs"] },
        { code: "replay-record-incomplete", sha256: null, reasons: ["prompt_omitted"] },
      ],
    }));

    expect(screen.getByText("No model here holds exactly its files")).toBeInTheDocument();
    expect(screen.getByText("It lists the same input picture twice.")).toBeInTheDocument();
    expect(screen.getByText("The prompt is not included.")).toBeInTheDocument();
    expect(screen.getByText(/The record leaves out something a replay needs/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Generate again" })).toBeNull();
  });

  it("offers a video made from a picture, which starts from the same picture", () => {
    renderReplay(plan({ operation: "image_to_video" }));

    expect(screen.getByRole("button", { name: "Generate again" })).toBeInTheDocument();
  });

  it("starts an edit of a picture the same way", async () => {
    replay.mockResolvedValue({});
    const onStarted = renderReplay(plan({ operation: "image_to_image" }));

    fireEvent.click(screen.getByRole("button", { name: "Generate again" }));

    await waitFor(() => expect(onStarted).toHaveBeenCalledWith("chat_again"));
    expect(replay).toHaveBeenCalledWith("chat_again", CONTENT);
  });

  it("does not offer to start a kind of generation a replay cannot start yet", () => {
    renderReplay(plan({ operation: "a_later_kind" }));

    expect(screen.getByText(/cannot be started again yet/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Generate again" })).toBeNull();
  });

  it("removes the empty chat and says where it would differ when the replay is refused", async () => {
    replay.mockRejectedValue(Object.assign(new Error("server words"), {
      code: "replay-differs",
      payload: { code: "replay-differs", sections: ["settings", "output_count"] },
    }));
    const onStarted = renderReplay(plan());

    fireEvent.click(screen.getByRole("button", { name: "Generate again" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "would not match the record exactly (it would differ in the settings and how many results it makes)",
    );
    expect(discard).toHaveBeenCalledWith("chat_again");
    expect(onStarted).not.toHaveBeenCalled();
    expect(screen.queryByText("server words")).toBeNull();
  });
});

describe("reading the replay plan", () => {
  it("refuses a plan that contradicts itself or names a malformed hash", () => {
    const refused = { code: "replay-model-missing", kind: "model", sha256: null, reasons: [] };
    expect(() => readReplayPlan({ ...plan(), refusals: [refused] })).toThrow("contradicts");
    expect(() => readReplayPlan({ ...plan(), ready: false, refusals: [] })).toThrow("contradicts");
    expect(() => readReplayPlan({ ...plan(), ready: false, refusals: [{ ...refused, sha256: "x" }] }))
      .toThrow("malformed");
    expect(readReplayPlan({ ...plan(), resolved: { profile_id: "local" } }).ready).toBe(true);
  });
});
