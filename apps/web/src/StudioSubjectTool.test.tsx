/** Replacing a subject cuts it out, then redraws only its place from a second picture. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { readCutoutMask } from "./studioBackground";
import { createMask, encodeMaskPng } from "./studioMasks";
import type { ChatDetail, Message, StudioToolCapability } from "./types";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: { favoriteArtifact: vi.fn(), artifact: vi.fn(), editTemplates: vi.fn(), studioCapabilities: vi.fn(), artifacts: vi.fn() },
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
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

const BRUSH_REASON = "Install an inpainting workflow to edit part of a picture.";
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

function tool(kind: StudioToolCapability["kind"], changes: Partial<StudioToolCapability> = {}) {
  return {
    kind,
    workflow_class: "image_to_image",
    available: true,
    reason: null,
    workflow_revision_id: null,
    adapter_asset_id: null,
    ...changes,
  };
}

async function openWith(subject: Partial<StudioToolCapability>) {
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({
    tools: [
      tool("isolate", { workflow_class: "matting", workflow_revision_id: "wfrev_cutout" }),
      tool("subject", { workflow_class: "reference_edit", workflow_revision_id: "wfrev_two", ...subject }),
      // Unavailable on purpose: its reason on the rail says the report has arrived.
      tool("brush", { workflow_class: "inpaint", available: false, reason: BRUSH_REASON }),
    ],
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  } as ReturnType<typeof useStudioImage>);
  const subjectAtCentre = createMask(400, 200);
  subjectAtCentre.data[100 * 400 + 200] = 255;
  vi.mocked(readCutoutMask).mockResolvedValue(subjectAtCentre);
  apply = vi.fn();
  session = { id: "chat-studio", messages: [] } as unknown as ChatDetail;
  mockSession();
  view = render(tree());
  fireEvent.click(screen.getByRole("button", { name: /^Replace the subject/ }));
  await screen.findByRole("button", { name: `Select part of the picture - ${BRUSH_REASON}` });
}

function nameTheSubject(name: string) {
  fireEvent.change(screen.getByLabelText("Name of the new subject"), { target: { value: name } });
}

function choosePicture() {
  const picture = new File(["neutral"], "new-subject.png", { type: "image/png" });
  fireEvent.change(screen.getByLabelText("Picture of the new subject"), {
    target: { files: [picture] },
  });
  return picture;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("cuts the subject out on Isolate's workflow, then redraws only its place from the picture", async () => {
  await openWith({});
  const replace = screen.getByRole("button", { name: "Replace subject" });
  // The report is in: only the picture and the new subject's name are missing.
  expect(replace).toHaveAttribute("aria-disabled", "true");
  const picture = choosePicture();
  expect(screen.getByText("new-subject.png")).toBeInTheDocument();
  expect(replace).toHaveAttribute("aria-disabled", "true");
  nameTheSubject("the dog");
  await waitFor(() => expect(replace).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(replace);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));

  const [cutWords, cutSource, cutMask, cutSettings, cutWorkflow, accepted] = apply.mock.calls[0];
  expect(cutWords).toBe("Cut the subject out onto a transparent background.");
  expect(cutSource).toBe("art-1");
  expect(cutMask).toBeUndefined();
  expect(cutSettings).toBeUndefined();
  expect(cutWorkflow).toBe("wfrev_cutout");

  accepted({ assistant_message: { id: "msg-cutout" } });
  showSession([message("pending")]);
  expect(apply).toHaveBeenCalledTimes(1);
  showSession([message("complete", "art-cutout")]);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(2));

  expect(readCutoutMask).toHaveBeenCalledWith("art-cutout", 400, 200);
  const [words, source, mask, settings, workflow, , second] = apply.mock.calls[1];
  expect(words).toBe(
    "Replace the subject with the dog from the second picture. Keep everything around it exactly as it is.",
  );
  // The picture being edited first, the new subject's picture after it, and
  // the result placed back into the first alone.
  expect(source).toBe("art-1");
  expect(second).toBe(picture);
  expect(mask).toMatchObject({ featherPx: 4, invert: false, apply: "blend", references: 1 });
  expect(settings).toBeUndefined();
  expect(workflow).toBe("wfrev_two");
  // The subject's own coverage, grown by its reach on a 400 by 200 picture
  // (8 pixels) so a new subject has room, and no further.
  const grown = vi.mocked(encodeMaskPng).mock.calls[0][0];
  expect(grown.data[100 * 400 + 208]).toBe(255);
  expect(grown.data[100 * 400 + 209]).toBe(0);
  expect(grown.data[92 * 400 + 200]).toBe(255);
  expect(grown.data[91 * 400 + 200]).toBe(0);

  // Another look at the same finished cutout never sends the redraw again.
  showSession([message("complete", "art-cutout")]);
  expect(apply).toHaveBeenCalledTimes(2);
});

it("takes the new subject from a picture the library holds, and sends that picture as it is", async () => {
  await openWith({});
  const item = (id: string, name: string) => ({
    id, sha256: "a".repeat(64), kind: "image", media_type: "image/png", size_bytes: 1, original_name: name,
    metadata_json: {}, created_at: "2026-01-01T00:00:00Z", reference_count: 0, chat_ids: [], project_ids: [],
  });
  vi.mocked(api.artifacts).mockResolvedValue([item("art-red", "red-cube.png"), item("art-blue", "blue-cube.png")]);

  fireEvent.click(screen.getByRole("button", { name: "Choose from the library" }));
  fireEvent.click(await screen.findByRole("button", { name: "red-cube.png" }));
  fireEvent.click(screen.getByRole("button", { name: "blue-cube.png" }));
  fireEvent.click(screen.getByRole("button", { name: "Use this picture" }));

  expect(screen.getByText("blue-cube.png")).toBeInTheDocument();
  nameTheSubject("the blue cube");
  const replace = screen.getByRole("button", { name: "Replace subject" });
  await waitFor(() => expect(replace).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(replace);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  apply.mock.calls[0][5]({ assistant_message: { id: "msg-cutout" } });
  showSession([message("complete", "art-cutout")]);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(2));

  // The library's own picture follows the source, named rather than sent again.
  expect(apply.mock.calls[1][6]).toBe("art-blue");
  expect(apply.mock.calls[1][2]).toMatchObject({ apply: "blend", references: 1 });
});

it("waits for the new subject's name before it replaces anything, and redraws what is named", async () => {
  await openWith({});
  choosePicture();
  const replace = screen.getByRole("button", { name: "Replace subject" });
  const name = screen.getByLabelText("Name of the new subject");
  expect(name).toHaveAttribute("aria-required", "true");
  expect(name).toHaveAccessibleDescription(/Replace subject waits for them/);
  // A picture alone, or a name of spaces, is not enough to run on.
  expect(replace).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(replace);
  nameTheSubject("   ");
  expect(replace).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(replace);
  expect(apply).not.toHaveBeenCalled();

  nameTheSubject("the blue cube");
  await waitFor(() => expect(replace).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(replace);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  apply.mock.calls[0][5]({ assistant_message: { id: "msg-cutout" } });
  showSession([message("complete", "art-cutout")]);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(2));

  expect(apply.mock.calls[1][0]).toBe(
    "Replace the subject with the blue cube from the second picture. Keep everything around it exactly as it is.",
  );
});

it("says what to install when nothing here reads a second picture", async () => {
  const reason =
    "Install an image editing workflow that takes a second picture to replace a subject with one from another picture.";
  await openWith({ available: false, reason, workflow_revision_id: null });

  expect(await screen.findByRole("status")).toHaveTextContent("takes a second picture");
  choosePicture();
  const replace = screen.getByRole("button", { name: "Replace subject" });
  expect(replace).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(replace);
  expect(apply).not.toHaveBeenCalled();
});
