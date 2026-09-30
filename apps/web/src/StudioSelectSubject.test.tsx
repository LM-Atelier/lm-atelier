/** Select the subject: offered beside the selection where a workflow can cut one out, and selecting it there. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { readCutoutMask } from "./studioBackground";
import { createMask, fillRect } from "./studioMasks";
import type { ChatDetail, Message, StudioToolCapability, TurnAccepted } from "./types";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession, type StudioStep } from "./useStudioSession";
import { SUBJECT_ELSEWHERE } from "./useStudioSubjectSelection";

vi.mock("./api", () => ({
  api: { favoriteArtifact: vi.fn(), artifact: vi.fn(), editTemplates: vi.fn(), studioCapabilities: vi.fn() },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./studioBackground", async (original) => ({
  ...(await original<typeof import("./studioBackground")>()),
  readCutoutMask: vi.fn(),
}));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

const source: StudioStep = {
  messageId: "source",
  artifactId: "art-1",
  instruction: "",
  beforeArtifactId: null,
  isSource: true,
  generationIdentity: null,
};
const earlier: StudioStep = {
  messageId: "answer-0",
  artifactId: "art-0",
  instruction: "warmer light",
  beforeArtifactId: "art-1",
  isSource: false,
  generationIdentity: null,
};
const cutout: StudioStep = {
  messageId: "msg-cutout",
  artifactId: "art-cutout",
  instruction: "Cut the subject out onto a transparent background.",
  beforeArtifactId: "art-1",
  isSource: false,
  generationIdentity: null,
};

let apply: ReturnType<typeof vi.fn>;
let client: QueryClient;
let view: ReturnType<typeof render>;

function isolate(available: boolean): StudioToolCapability {
  return {
    kind: "isolate",
    workflow_class: "matting",
    available,
    reason: available ? null : "No workflow here can cut a subject out.",
    workflow_revision_id: available ? "wfrev_cutout" : null,
    adapter_asset_id: null,
  };
}

function showSession(steps: StudioStep[], messages: Message[] = []) {
  vi.mocked(useStudioSession).mockReturnValue({
    steps,
    previewArtifactId: null,
    sessionId: "chat-studio",
    session: { id: "chat-studio", messages } as unknown as ChatDetail,
    busy: false,
    error: null,
    apply,
  } as unknown as ReturnType<typeof useStudioSession>);
}

function tree() {
  return (
    <QueryClientProvider client={client}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>
  );
}

async function open(steps: StudioStep[], canCutOut = true) {
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [isolate(canCutOut)] });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  // One decoded picture per artifact, as the real hook keeps.
  const pictures = new Map<string, ImageBitmap>();
  vi.mocked(useStudioImage).mockImplementation((artifactId: string | null) => {
    if (artifactId && !pictures.has(artifactId)) {
      pictures.set(artifactId, { width: 40, height: 20, close: vi.fn() } as unknown as ImageBitmap);
    }
    return { bitmap: artifactId ? pictures.get(artifactId)! : null, error: null, reload: vi.fn() };
  });
  apply = vi.fn();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  showSession(steps);
  view = render(tree());
  await waitFor(() => expect(api.studioCapabilities).toHaveBeenCalled());
  fireEvent.click(screen.getByRole("button", { name: "Select part of the picture" }));
}

function answerTheCutout() {
  const accepted = apply.mock.calls[0][5] as (accepted: TurnAccepted) => void;
  act(() => accepted({ assistant_message: { id: "msg-cutout" } } as unknown as TurnAccepted));
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("selects the subject the cutout finds, on the picture it was asked for, in place of the selection", async () => {
  const subject = createMask(40, 20);
  fillRect(subject, 0, 0, 20, 20);
  vi.mocked(readCutoutMask).mockResolvedValue(subject);
  await open([source]);

  fireEvent.click(await screen.findByRole("button", { name: "Select the subject" }));

  expect(apply).toHaveBeenCalledTimes(1);
  const [words, picture, mask, settings, workflow] = apply.mock.calls[0];
  expect([words, picture, mask, settings, workflow]).toEqual([
    "Cut the subject out onto a transparent background.", "art-1", undefined, undefined, "wfrev_cutout",
  ]);
  answerTheCutout();
  showSession([source], [{ id: "msg-cutout", role: "assistant", status: "pending", parts: [] } as unknown as Message]);
  view.rerender(tree());
  expect(screen.getByText("Finding the subject…")).toBeInTheDocument();

  // The cutout joins the strip as the newest result, and the canvas stays where the subject was asked for.
  showSession([source, cutout], [{
    id: "msg-cutout", role: "assistant", status: "complete",
    parts: [{ type: "image", artifact_id: "art-cutout", metadata_json: {} }],
  } as unknown as Message]);
  view.rerender(tree());

  await waitFor(() => expect(screen.getByText("50.0% of the image selected")).toBeInTheDocument());
  expect(readCutoutMask).toHaveBeenCalledWith("art-cutout", 40, 20);
  const strip = screen.getByRole("group", { name: "Edit history" });
  expect(within(strip).getByRole("button", { pressed: true })).toHaveAccessibleName(/^The original image/);
  expect(screen.queryByText("Finding the subject…")).toBeNull();

  // A selection like any other: undone to what was there before.
  fireEvent.click(screen.getByRole("button", { name: "Undo the selection change" }));
  expect(screen.getByText("Nothing selected yet - paint over what you want to change.")).toBeInTheDocument();
});

it("leaves the selection as it was when the canvas has moved to another picture", async () => {
  const subject = createMask(40, 20);
  fillRect(subject, 0, 0, 20, 20);
  vi.mocked(readCutoutMask).mockResolvedValue(subject);
  await open([source, earlier]);
  fireEvent.click(await screen.findByRole("button", { name: "Select the subject" }));
  expect(apply.mock.calls[0][1]).toBe("art-0");
  answerTheCutout();

  const strip = screen.getByRole("group", { name: "Edit history" });
  fireEvent.click(within(strip).getByRole("button", { name: /^The original image/ }));
  showSession([source, earlier, { ...cutout, beforeArtifactId: "art-0" }], [{
    id: "msg-cutout", role: "assistant", status: "complete",
    parts: [{ type: "image", artifact_id: "art-cutout", metadata_json: {} }],
  } as unknown as Message]);
  view.rerender(tree());

  expect(await screen.findByRole("alert")).toHaveTextContent(SUBJECT_ELSEWHERE);
  expect(screen.getByText("Nothing selected yet - paint over what you want to change.")).toBeInTheDocument();
});

it("is not offered where no workflow can cut a subject out", async () => {
  await open([source], false);

  expect(await screen.findByRole("button", { name: "Invert" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Select the subject" })).toBeNull();
});
