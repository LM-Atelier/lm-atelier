/** Replacing a subject removes the old one and places the new one, cut out of a second picture, where it stood. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { readCutoutMask } from "./studioBackground";
import { createMask, encodeMaskPng } from "./studioMasks";
import { encodeRgbaPng, readCutoutPixels } from "./studioPlaceSubject";
import type { ChatDetail, Message, StudioToolCapability } from "./types";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession } from "./useStudioSession";
import { NO_NEW_SUBJECT, NO_SUBJECT_FOUND, REMOVAL_FAILED } from "./useStudioSubjectPlace";

vi.mock("./api", () => ({
  api: {
    favoriteArtifact: vi.fn(),
    artifact: vi.fn(),
    editTemplates: vi.fn(),
    studioCapabilities: vi.fn(),
    artifacts: vi.fn(),
    upload: vi.fn(),
  },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./studioBackground", async (original) => ({
  ...(await original<typeof import("./studioBackground")>()),
  readCutoutMask: vi.fn(),
}));
vi.mock("./studioMasks", async (original) => ({
  ...(await original<typeof import("./studioMasks")>()),
  encodeMaskPng: vi.fn(async () => new Blob(["old subject"], { type: "image/png" })),
}));
vi.mock("./studioPlaceSubject", async (original) => ({
  ...(await original<typeof import("./studioPlaceSubject")>()),
  readCutoutPixels: vi.fn(),
  encodeRgbaPng: vi.fn(async () => new Blob(["new subject"], { type: "image/png" })),
}));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

const BRUSH_REASON = "Install an inpainting workflow to edit part of a picture.";
const CUT_WORDS = "Cut the subject out onto a transparent background.";
const REMOVE_WORDS =
  "Remove the subject. Fill the space it leaves to match what surrounds it, and leave everything else unchanged.";
const GREEN = [0, 160, 0, 255];
let apply: ReturnType<typeof vi.fn>;
let localEdit: ReturnType<typeof vi.fn>;
let session: ChatDetail;
let view: ReturnType<typeof render>;

function message(id: string, status: Message["status"], artifactId?: string): Message {
  return {
    id,
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
    localEdit,
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

/** The old subject: solid from (300, 100) to (339, 179), off to the right of a 400 by 200 picture, with a faint edge. */
function oldSubject() {
  const mask = createMask(400, 200);
  for (let y = 100; y < 180; y += 1) {
    for (let x = 300; x < 340; x += 1) mask.data[y * 400 + x] = 255;
  }
  mask.data[99 * 400 + 320] = 60;
  return mask;
}

/** The new subject's cutout: a green figure ten by twenty, in the middle of its own picture. */
function newSubject() {
  const data = new Uint8ClampedArray(30 * 40 * 4);
  for (let y = 10; y < 30; y += 1) {
    for (let x = 10; x < 20; x += 1) data.set(GREEN, (y * 30 + x) * 4);
  }
  return { width: 30, height: 40, data };
}

async function openWith(subject: Partial<StudioToolCapability>) {
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.upload).mockResolvedValue({ id: "art-new-picture" } as never);
  vi.mocked(api.studioCapabilities).mockResolvedValue({
    tools: [
      tool("isolate", { workflow_class: "matting", workflow_revision_id: "wfrev_cutout" }),
      tool("subject", { workflow_class: "matting", workflow_revision_id: "wfrev_cutout", ...subject }),
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
  vi.mocked(readCutoutMask).mockResolvedValue(oldSubject());
  vi.mocked(readCutoutPixels).mockResolvedValue(newSubject());
  apply = vi.fn();
  localEdit = vi.fn();
  session = { id: "chat-studio", messages: [] } as unknown as ChatDetail;
  mockSession();
  view = render(tree());
  fireEvent.click(screen.getByRole("button", { name: /^Replace the subject/ }));
  await screen.findByRole("button", { name: `Select part of the picture - ${BRUSH_REASON}` });
}

function choosePicture() {
  const picture = new File(["neutral"], "new-subject.png", { type: "image/png" });
  fireEvent.change(screen.getByLabelText("Picture of the new subject"), {
    target: { files: [picture] },
  });
  return picture;
}

/** Press Replace subject, and let the old subject be found and the new one's picture taken. */
async function replaceUpToTheNewCutout() {
  const replace = screen.getByRole("button", { name: "Replace subject" });
  await waitFor(() => expect(replace).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(replace);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  apply.mock.calls[0][5]({ assistant_message: { id: "msg-find" } });
  showSession([message("msg-find", "complete", "art-found")]);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(2));
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("removes the old subject and places the new one, cut out of a chosen picture, where the old one stood", async () => {
  await openWith({});
  const replace = screen.getByRole("button", { name: "Replace subject" });
  // The report is in: only the new subject's picture is missing, and nothing needs naming.
  expect(replace).toHaveAttribute("aria-disabled", "true");
  expect(screen.queryByLabelText("Name of the new subject")).not.toBeInTheDocument();
  const picture = choosePicture();
  expect(screen.getByText("new-subject.png")).toBeInTheDocument();
  await replaceUpToTheNewCutout();

  // First the old subject is found, on Isolate's workflow.
  const [findWords, findSource, findMask, findSettings, findWorkflow] = apply.mock.calls[0];
  expect([findWords, findSource, findMask, findSettings, findWorkflow]).toEqual([
    CUT_WORDS, "art-1", undefined, undefined, "wfrev_cutout",
  ]);
  expect(readCutoutMask).toHaveBeenCalledWith("art-found", 400, 200);
  // Then the new subject is cut out of its own picture, uploaded first.
  expect(api.upload).toHaveBeenCalledWith(picture);
  const [cutWords, cutSource, cutMask, , cutWorkflow] = apply.mock.calls[1];
  expect([cutWords, cutSource, cutMask, cutWorkflow]).toEqual([CUT_WORDS, "art-new-picture", undefined, "wfrev_cutout"]);
  apply.mock.calls[1][5]({ assistant_message: { id: "msg-cut" } });
  showSession([message("msg-find", "complete", "art-found"), message("msg-cut", "complete", "art-cut")]);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(3));

  // Only then is the old one removed: the source again, through what the old
  // subject covered, grown by its reach (8 pixels on a 400 by 200 picture),
  // softened by half that, on the studio's own workflow.
  expect(readCutoutPixels).toHaveBeenCalledWith("art-cut");
  const [words, source, mask, settings, workflow] = apply.mock.calls[2];
  expect(words).toBe(REMOVE_WORDS);
  expect(source).toBe("art-1");
  expect(mask).toMatchObject({ featherPx: 4, invert: false, apply: "blend" });
  expect(mask.references).toBeUndefined();
  expect(settings).toBeUndefined();
  expect(workflow).toBeUndefined();
  const grown = vi.mocked(encodeMaskPng).mock.calls[0][0];
  expect(grown.data[120 * 400 + 347]).toBe(255);
  expect(grown.data[120 * 400 + 348]).toBe(0);
  expect(grown.data[120 * 400 + 291]).toBe(0);
  expect(grown.data[120 * 400 + 292]).toBe(255);

  // The new subject fills the old one's box, standing where it stood, and
  // nothing is drawn in the middle of the picture, where a redraw put it.
  const [width, height, placed] = vi.mocked(encodeRgbaPng).mock.calls[0];
  expect([width, height]).toEqual([400, 200]);
  const at = (x: number, y: number) => Array.from(placed.slice((y * 400 + x) * 4, (y * 400 + x) * 4 + 4));
  expect(at(302, 102)).toEqual(GREEN);
  expect(at(337, 177)).toEqual(GREEN);
  // Its outermost pixels are its soft edge, still its own colour.
  expect(at(300, 100).slice(0, 3)).toEqual([0, 160, 0]);
  expect(at(299, 100)[3]).toBe(0);
  expect(at(320, 99)[3]).toBe(0);
  expect(at(200, 100)[3]).toBe(0);

  apply.mock.calls[2][5]({ assistant_message: { id: "msg-remove" } });
  showSession([
    message("msg-find", "complete", "art-found"),
    message("msg-cut", "complete", "art-cut"),
    message("msg-remove", "complete", "art-removed"),
  ]);
  await waitFor(() => expect(localEdit).toHaveBeenCalledTimes(1));

  // Laid over the picture the old one was removed from, with no model.
  const [operation, removed, done, details] = localEdit.mock.calls[0];
  expect(operation).toBe("subject");
  expect(removed).toBe("art-removed");
  expect(details).toEqual({ subject: { placed: await vi.mocked(encodeRgbaPng).mock.results[0].value } });
  expect(screen.getByRole("button", { name: "Applying…" })).toBeInTheDocument();
  done();
  expect(await screen.findByRole("button", { name: "Replace subject" })).toBeInTheDocument();

  // Another look at the same finished steps never repeats any of them.
  showSession([
    message("msg-find", "complete", "art-found"),
    message("msg-cut", "complete", "art-cut"),
    message("msg-remove", "complete", "art-removed"),
  ]);
  expect(apply).toHaveBeenCalledTimes(3);
  expect(localEdit).toHaveBeenCalledTimes(1);
});

it("cuts the new subject out of a picture the library holds, without sending it again", async () => {
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
  await replaceUpToTheNewCutout();

  expect(apply.mock.calls[1][1]).toBe("art-blue");
  expect(api.upload).not.toHaveBeenCalled();
});

it("leaves the picture as it was when no subject is found in it", async () => {
  await openWith({});
  vi.mocked(readCutoutMask).mockResolvedValue(createMask(400, 200));
  choosePicture();
  const replace = screen.getByRole("button", { name: "Replace subject" });
  await waitFor(() => expect(replace).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(replace);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  apply.mock.calls[0][5]({ assistant_message: { id: "msg-find" } });
  showSession([message("msg-find", "complete", "art-found")]);

  expect(await screen.findByText(NO_SUBJECT_FOUND)).toBeInTheDocument();
  expect(apply).toHaveBeenCalledTimes(1);
  expect(api.upload).not.toHaveBeenCalled();
});

it("leaves the picture as it was when the second picture holds no subject", async () => {
  await openWith({});
  vi.mocked(readCutoutPixels).mockResolvedValue({ width: 30, height: 40, data: new Uint8ClampedArray(30 * 40 * 4) });
  choosePicture();
  await replaceUpToTheNewCutout();
  apply.mock.calls[1][5]({ assistant_message: { id: "msg-cut" } });
  showSession([message("msg-find", "complete", "art-found"), message("msg-cut", "complete", "art-cut")]);

  expect(await screen.findByText(NO_NEW_SUBJECT)).toBeInTheDocument();
  // Nothing was removed.
  expect(apply).toHaveBeenCalledTimes(2);
});

it("says so when the old subject could not be removed, and places nothing", async () => {
  await openWith({});
  choosePicture();
  await replaceUpToTheNewCutout();
  apply.mock.calls[1][5]({ assistant_message: { id: "msg-cut" } });
  showSession([message("msg-find", "complete", "art-found"), message("msg-cut", "complete", "art-cut")]);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(3));
  apply.mock.calls[2][5]({ assistant_message: { id: "msg-remove" } });
  showSession([
    message("msg-find", "complete", "art-found"),
    message("msg-cut", "complete", "art-cut"),
    message("msg-remove", "failed"),
  ]);

  expect(await screen.findByText(REMOVAL_FAILED)).toBeInTheDocument();
  expect(localEdit).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Replace subject" })).toBeInTheDocument();
});

it("says what to install when nothing here can edit a picture", async () => {
  const reason = "Install an image editing workflow to change a picture.";
  await openWith({ available: false, reason });

  expect(await screen.findByRole("status")).toHaveTextContent(reason);
  choosePicture();
  const replace = screen.getByRole("button", { name: "Replace subject" });
  expect(replace).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(replace);
  expect(apply).not.toHaveBeenCalled();
});
