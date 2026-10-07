/** Replacing a background cuts the subject out, then redraws only what surrounds it. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { readCutoutMask } from "./studioBackground";
import { createMask } from "./studioMasks";
import type { ChatDetail, Message, StudioToolCapability } from "./types";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: { favoriteArtifact: vi.fn(), artifact: vi.fn(), editTemplates: vi.fn(), studioCapabilities: vi.fn() },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./studioBackground", async (original) => ({
  ...(await original<typeof import("./studioBackground")>()),
  readCutoutMask: vi.fn(),
}));
vi.mock("./studioMasks", async (original) => ({
  ...(await original<typeof import("./studioMasks")>()),
  encodeMaskPng: vi.fn(async () => new Blob(["subject"], { type: "image/png" })),
}));
// The studio's own workflow can run, so the redraw has somewhere to go.
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

const SCENE = "a quiet beach at sunset";
let apply: ReturnType<typeof vi.fn>;
let session: ChatDetail;
let view: ReturnType<typeof render>;

function message(status: Message["status"], artifactId?: string): Message {
  return {
    id: "msg-cutout",
    role: "assistant",
    status,
    parts: artifactId ? [{ type: "image", artifact_id: artifactId, metadata_json: {} }] : [],
  } as unknown as Message;
}

function mockSession() {
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [{ artifactId: "art-1", instruction: null, generationIdentity: null }],
    previewArtifactId: null,
    sessionId: "chat-studio",
    session,
    busy: false,
    error: null,
    apply,
  } as unknown as ReturnType<typeof useStudioSession>);
}

function tree() {
  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>
  );
}

function showSession(messages: Message[]) {
  session = { id: "chat-studio", messages } as unknown as ChatDetail;
  mockSession();
  view.rerender(tree());
}

function openWith(background: Partial<StudioToolCapability>) {
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({
    tools: [
      {
        kind: "background",
        workflow_class: "matting",
        available: true,
        reason: null,
        workflow_revision_id: "wfrev_cutout",
        adapter_asset_id: null,
        ...background,
      },
    ],
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  } as ReturnType<typeof useStudioImage>);
  vi.mocked(readCutoutMask).mockResolvedValue(createMask(400, 200));
  apply = vi.fn();
  session = { id: "chat-studio", messages: [] } as unknown as ChatDetail;
  mockSession();
  view = render(tree());
  fireEvent.click(screen.getByRole("button", { name: "Replace the background" }));
}

async function startReplacing() {
  fireEvent.change(screen.getByRole("textbox"), { target: { value: SCENE } });
  const replace = await screen.findByRole("button", { name: "Replace background" });
  await waitFor(() => expect(replace).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(replace);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  return replace;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("cuts the subject out first, then redraws everything around it from the words", async () => {
  openWith({});
  expect(screen.queryByRole("button", { name: "Invert" })).toBeNull();
  await startReplacing();

  const [cutWords, cutSource, cutMask, cutSettings, cutWorkflow, accepted] = apply.mock.calls[0];
  expect(cutWords).toBe("Cut the subject out onto a transparent background.");
  expect(cutSource).toBe("art-1");
  expect(cutMask).toBeUndefined();
  expect(cutSettings).toBeUndefined();
  // The cutout runs on the workflow the report names, as Isolate does.
  expect(cutWorkflow).toBe("wfrev_cutout");

  accepted({ assistant_message: { id: "msg-cutout" } });
  showSession([message("pending")]);
  expect(readCutoutMask).not.toHaveBeenCalled();
  expect(apply).toHaveBeenCalledTimes(1);
  // The redraw is sent from here once the cutout is done, so closing waits for it.
  expect(screen.getByRole("button", { name: "Close" })).toHaveAttribute("aria-disabled", "true");

  showSession([message("complete", "art-cutout")]);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(2));
  // Read at the picture's own size, so the subject lines up with it.
  expect(readCutoutMask).toHaveBeenCalledWith("art-cutout", 400, 200);
  const [words, source, mask, settings, workflow] = apply.mock.calls[1];
  expect(words).toBe(
    `Replace the background with ${SCENE}. Keep the subject exactly as it is, in the same place, size and pose.`,
  );
  // The original picture, not the cutout, with the subject's selection inverted
  // and placed back through the blend, so the subject keeps its own pixels.
  expect(source).toBe("art-1");
  expect(mask).toMatchObject({ featherPx: 0, invert: true, apply: "blend" });
  expect(mask.blob).toBeInstanceOf(Blob);
  expect(settings).toBeUndefined();
  expect(workflow).toBeUndefined();

  // Another look at the same finished cutout never sends the redraw again.
  showSession([message("complete", "art-cutout")]);
  expect(apply).toHaveBeenCalledTimes(2);
});

it("says so and sends nothing more when the cutout fails", async () => {
  openWith({});
  const replace = await startReplacing();
  apply.mock.calls[0][5]({ assistant_message: { id: "msg-cutout" } });
  showSession([message("failed")]);

  expect(await screen.findByText(/could not be cut out/)).toBeInTheDocument();
  await waitFor(() => expect(replace).toHaveAttribute("aria-disabled", "false"));
  expect(readCutoutMask).not.toHaveBeenCalled();
  expect(apply).toHaveBeenCalledTimes(1);
});

it("stops waiting when the cutout is refused", async () => {
  openWith({});
  const replace = await startReplacing();
  expect(replace).toHaveAttribute("aria-disabled", "true");
  apply.mock.calls[0][7]();
  view.rerender(tree());
  await waitFor(() => expect(replace).toHaveAttribute("aria-disabled", "false"));
});

it("says what to install when nothing here can cut a subject out", async () => {
  openWith({
    available: false,
    reason: "Install a background removal workflow to cut a subject out of a picture.",
    workflow_revision_id: null,
  });

  // The guidance, once the report has answered; before it, the Studio says it is checking.
  await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Install a background removal workflow"));
  fireEvent.change(screen.getByRole("textbox"), { target: { value: SCENE } });
  const replace = screen.getByRole("button", { name: "Replace background" });
  expect(replace).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(replace);
  expect(apply).not.toHaveBeenCalled();
});
