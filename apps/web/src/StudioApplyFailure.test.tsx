/** An edit that failed says so in the Studio, rather than vanishing with its words. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import type { ChatDetail, Message, TurnAccepted } from "./types";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession, type StudioStep } from "./useStudioSession";

vi.mock("./api", () => ({
  api: { favoriteArtifact: vi.fn(), artifact: vi.fn(), editTemplates: vi.fn(), studioCapabilities: vi.fn() },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

const source: StudioStep = {
  messageId: "source",
  artifactId: "art-source",
  instruction: "",
  beforeArtifactId: null,
  isSource: true,
  generationIdentity: null,
};

function turn(id: string, words: string): Message {
  return { id, role: "user", status: "complete", parts: [{ type: "text", text: words, metadata_json: {} }] } as unknown as Message;
}

function answer(id: string, status: Message["status"], reason?: string): Message {
  return {
    id,
    role: "assistant",
    status,
    parts: reason === undefined ? [] : [{ type: "error", text: reason, metadata_json: {} }],
  } as unknown as Message;
}

let client: QueryClient;
let view: ReturnType<typeof render>;
let apply: ReturnType<typeof vi.fn>;

function tree() {
  return (
    <QueryClientProvider client={client}>
      <StudioView sourceArtifactId="art-source" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>
  );
}

function showSession(messages: Message[]) {
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [source],
    previewArtifactId: null,
    sessionId: "chat-studio",
    session: { id: "chat-studio", messages } as unknown as ChatDetail,
    busy: messages.some((message) => message.status === "pending"),
    error: null,
    apply,
  } as unknown as ReturnType<typeof useStudioSession>);
}

function open(messages: Message[]) {
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-source", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
  vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  apply = vi.fn();
  showSession(messages);
  view = render(tree());
}

function show(messages: Message[]) {
  showSession(messages);
  view.rerender(tree());
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("says why the newest edit did not finish, until a newer edit is under way", () => {
  const failed = [turn("turn-1", "Make it warmer"), answer("answer-1", "failed", "The workflow stopped at its sampler.")];
  open(failed);

  expect(screen.getByRole("alert")).toHaveTextContent("The edit did not finish: The workflow stopped at its sampler.");
  // Not sent from this visit, so there are no words of its own to offer back.
  expect(screen.queryByRole("button", { name: "Use the words again" })).toBeNull();

  show([...failed, turn("turn-2", "Make it cooler"), answer("answer-2", "pending")]);

  expect(screen.queryByRole("alert")).toBeNull();
});

it("says so again for a later failure with the same reason, once the first was dismissed", () => {
  const first = [turn("turn-1", "Make it warmer"), answer("answer-1", "failed", "Out of memory")];
  open(first);
  fireEvent.click(screen.getByRole("button", { name: "Dismiss error" }));
  expect(screen.queryByRole("alert")).toBeNull();

  show([...first, turn("turn-2", "Make it warmer"), answer("answer-2", "failed", "Out of memory")]);

  expect(screen.getByRole("alert")).toHaveTextContent("The edit did not finish: Out of memory");
});

it("offers the failed edit's words again, since they were cleared when it was taken", async () => {
  open([]);
  apply.mockImplementation((...args: unknown[]) =>
    (args[5] as (accepted: TurnAccepted) => void)(
      { user_message: { id: "turn-1" }, assistant_message: { id: "answer-1" } } as unknown as TurnAccepted,
    ));
  const words = screen.getByRole("textbox");
  fireEvent.change(words, { target: { value: "Make it warmer" } });
  const send = screen.getByRole("button", { name: "Apply edit" });
  await waitFor(() => expect(send).toHaveAttribute("aria-disabled", "false"));

  fireEvent.click(send);
  expect(apply).toHaveBeenCalledTimes(1);
  expect(words).toHaveValue("");

  show([turn("turn-1", "Make it warmer"), answer("answer-1", "failed", "Out of memory")]);
  fireEvent.click(screen.getByRole("button", { name: "Use the words again" }));

  expect(screen.getByRole("textbox")).toHaveValue("Make it warmer");
});
