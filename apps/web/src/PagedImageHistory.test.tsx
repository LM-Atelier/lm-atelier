import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ArtifactPart } from "./ArtifactPart";
import { api } from "./api";
import type { ChatEditLineagePage, MessagePart } from "./types";

vi.mock("./api", () => ({ api: { chatEditLineage: vi.fn() } }));
beforeEach(() => vi.mocked(api.chatEditLineage).mockReset());
afterEach(cleanup);

const part: MessagePart = { id: "output-part", position: 0, type: "image", artifact_id: "output",
  text: null, metadata_json: {} };
const latest: ChatEditLineagePage = { chat_id: "chat-one", result_message_id: "result-one",
  steps: [
    { artifact_id: "second-input", message_id: "user-3", instruction: "Make it blue" },
    { artifact_id: "first-input", message_id: "user-2", instruction: "Soften the edges" },
  ], next_before: "user-2" };

function setup(resultId = "result-one") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  const view = (id: string) => <QueryClientProvider client={client}>
    <ArtifactPart part={part} origin="edited" editHistory={{ chatId: "chat-one", resultId: id }} />
  </QueryClientProvider>;
  return { ...render(view(resultId)), view };
}

it("keeps off-page comparisons and pages older edits with retry and retained focus", async () => {
  vi.mocked(api.chatEditLineage).mockResolvedValueOnce(latest).mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValueOnce({ ...latest, steps: [
      { artifact_id: "original", message_id: "user-1", instruction: "Adjust the lighting" },
    ], next_before: null });
  setup();

  fireEvent.click(await screen.findByRole("button", { name: "Compare with the source" }));
  expect(screen.getByAltText("The source before the edit")).toHaveAttribute("src", "/api/artifacts/second-input/content");
  fireEvent.click(screen.getByRole("button", { name: "Close comparison" }));
  fireEvent.click(screen.getByRole("button", { name: "Show the edit lineage" }));
  expect(screen.getByText("Latest 2 steps")).toBeVisible();
  const older = screen.getByRole("button", { name: "Load older edits" });
  older.focus();
  fireEvent.click(older);
  await screen.findByText("Older edits could not be loaded.");
  expect(screen.getByText("Make it blue")).toBeVisible();
  expect(older).toHaveFocus();
  fireEvent.click(older);
  await screen.findByText("Adjust the lighting");
  expect(screen.getByText("3 steps")).toBeVisible();
  expect(older).toHaveTextContent("All edits loaded");
  expect(older).toHaveFocus();
  expect(vi.mocked(api.chatEditLineage).mock.calls.map((call) => call[2]?.limit)).toEqual([2, 40, 40]);
  expect(vi.mocked(api.chatEditLineage).mock.calls.slice(1).every((call) => call[2]?.before === "user-2")).toBe(true);
});

it("offers a retry when the source read fails without hiding the result", async () => {
  vi.mocked(api.chatEditLineage).mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValueOnce({ ...latest, steps: latest.steps.slice(0, 1), next_before: null });
  setup();
  fireEvent.click(await screen.findByRole("button", { name: "Retry image history" }));
  await screen.findByRole("button", { name: "Compare with the source" });
  expect(screen.getByAltText("Edited image")).toBeVisible();
  expect(screen.queryByRole("button", { name: "Show the edit lineage" })).toBeNull();
});

it("cancels an old result read and ignores its late response", async () => {
  let finish: (value: ChatEditLineagePage) => void = () => undefined;
  let signal: AbortSignal | undefined;
  vi.mocked(api.chatEditLineage).mockImplementationOnce((_chat, _result, options) => {
    signal = options?.signal;
    return new Promise((resolve) => { finish = resolve; });
  }).mockResolvedValueOnce({ ...latest, result_message_id: "result-two", steps: [
    { artifact_id: "new-source", message_id: "new-user", instruction: "Add contrast" },
  ], next_before: null });
  const { rerender, view } = setup();
  await waitFor(() => expect(signal).toBeDefined());
  rerender(view("result-two"));
  fireEvent.click(await screen.findByRole("button", { name: "Compare with the source" }));
  expect(signal?.aborted).toBe(true);
  await act(async () => { finish(latest); });
  expect(screen.getByAltText("The source before the edit")).toHaveAttribute("src", "/api/artifacts/new-source/content");
});

it("hides absent edit controls and never requests history for previews", async () => {
  vi.mocked(api.chatEditLineage).mockResolvedValue({ ...latest, steps: [], next_before: null });
  const { rerender, view } = setup();
  await waitFor(() => expect(api.chatEditLineage).toHaveBeenCalledTimes(1));
  expect(screen.queryByRole("button", { name: "Compare with the source" })).toBeNull();
  rerender(<ArtifactPart part={{ ...part, metadata_json: { preview: true } }} origin="generated"
    editHistory={{ chatId: "chat-one", resultId: "preview" }} />);
  expect(api.chatEditLineage).toHaveBeenCalledTimes(1);
  rerender(view("result-one"));
});
