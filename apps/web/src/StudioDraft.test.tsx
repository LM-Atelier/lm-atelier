/** What the Studio holds on to while the person looks at another view, and gives back when they return. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { savedStudioDraftWords } from "./studioSavedDraft";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: {
    favoriteArtifact: vi.fn(),
    artifact: vi.fn().mockResolvedValue({ id: "art-1", favorite: false }),
    editTemplates: vi.fn().mockResolvedValue([]),
    studioCapabilities: vi.fn().mockResolvedValue({ tools: [] }),
  },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({ StudioWorkflowSelector: () => <div>Workflow chooser</div> }));

const NOTHING = "Nothing selected yet - paint over what you want to change.";

function selection(): string | null {
  return document.querySelector(".studio-selection-controls small")?.textContent ?? null;
}

/** The Studio on one picture of one session, with the app's one query client; the newest step is on the canvas. */
function open(client: QueryClient, sessionId = "chat-studio", artifactId = "art-1", later: string[] = []) {
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [artifactId, ...later].map((id) => ({ artifactId: id, instruction: null, generationIdentity: null })),
    previewArtifactId: null,
    sessionId,
    busy: false,
    error: null,
    apply: vi.fn(),
  } as unknown as ReturnType<typeof useStudioSession>);
  return render(
    <QueryClientProvider client={client}>
      <StudioView sourceArtifactId={artifactId} onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
}

/** Brush a short stroke and write some words. */
function work() {
  fireEvent.click(screen.getByRole("button", { name: "Select part of the picture" }));
  fireEvent.click(screen.getByRole("button", { name: "Brush a selection" }));
  const canvas = screen.getByRole("application");
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.keyDown(canvas, { key: "ArrowRight" });
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "make the sky warmer" } });
}

beforeEach(() => {
  // The canvas has no 2D context here, so it paints nothing; the selection and
  // the gestures that change it still run exactly as in a browser.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  } as ReturnType<typeof useStudioImage>);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("gives back the tool, its selection and the words after the person looks elsewhere", () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const visit = open(client);
  work();
  const drawn = selection();
  expect(drawn).not.toBe(NOTHING);

  visit.unmount();
  open(client);

  expect(screen.getByRole("button", { name: "Brush a selection" })).toHaveAttribute("aria-pressed", "true");
  expect(selection()).toBe(drawn);
  expect(screen.getByRole("textbox")).toHaveValue("make the sky warmer");
});

it("starts another session's picture clean", () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const visit = open(client);
  work();

  visit.unmount();
  open(client, "chat-other", "art-2");

  expect(screen.getByRole("button", { name: "Instruct the whole image" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("textbox")).toHaveValue("");
});

it("keeps what the latest visit left, even when it cleared what it was given back", () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const first = open(client);
  work();
  first.unmount();
  const second = open(client);
  // The second visit clears what it was given back, then leaves.
  fireEvent.click(screen.getByRole("button", { name: "Clear" }));
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "" } });
  second.unmount();

  open(client);

  expect(selection()).toBe(NOTHING);
  expect(screen.getByRole("textbox")).toHaveValue("");
});

it("gives the words back but not the selection when another picture of the session is on the canvas", () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const visit = open(client);
  work();

  visit.unmount();
  // An edit finished while the person was away, so its result is on the canvas now.
  open(client, "chat-studio", "art-1", ["art-2"]);

  expect(screen.getByRole("button", { name: "Instruct the whole image" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("textbox")).toHaveValue("make the sky warmer");
});

it("keeps a waiting draft through a visit that ends before its picture loads", () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const first = open(client);
  work();
  const drawn = selection();
  first.unmount();

  vi.mocked(useStudioImage).mockReturnValue({ bitmap: null, error: null, reload: vi.fn() } as ReturnType<typeof useStudioImage>);
  open(client).unmount();
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  } as ReturnType<typeof useStudioImage>);
  open(client);

  expect(selection()).toBe(drawn);
});

it("gives back the tool, its selection and the words after a reload, from what this browser wrote down", () => {
  const visit = open(new QueryClient({ defaultOptions: { queries: { retry: false } } }));
  work();
  const drawn = selection();
  visit.unmount();

  // A reload keeps nothing in memory: only what was written down as the Studio was left.
  open(new QueryClient({ defaultOptions: { queries: { retry: false } } }));

  expect(screen.getByRole("button", { name: "Brush a selection" })).toHaveAttribute("aria-pressed", "true");
  expect(selection()).toBe(drawn);
  expect(screen.getByRole("textbox")).toHaveValue("make the sky warmer");
});

it("gives the words back after a reload even though the picture is chosen after the Studio opens", () => {
  const visit = open(new QueryClient({ defaultOptions: { queries: { retry: false } } }));
  work();
  const drawn = selection();
  visit.unmount();

  // After a reload the Studio opens with no picture, and the session is known only once one is chosen.
  // With no picture there is nothing decoded; the chosen one arrives as a picture of its own.
  const picture = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
  vi.mocked(useStudioImage).mockImplementation((artifactId: string | null) => (
    { bitmap: artifactId ? picture : null, error: null, reload: vi.fn() } as ReturnType<typeof useStudioImage>
  ));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [], previewArtifactId: null, sessionId: null, busy: false, error: null, apply: vi.fn(),
  } as unknown as ReturnType<typeof useStudioSession>);
  const studio = (source: string | null) => (
    <QueryClientProvider client={client}>
      <StudioView sourceArtifactId={source} onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>
  );
  const view = render(studio(null));
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [{ artifactId: "art-1", instruction: null, generationIdentity: null }],
    previewArtifactId: null, sessionId: "chat-studio", busy: false, error: null, apply: vi.fn(),
  } as unknown as ReturnType<typeof useStudioSession>);
  view.rerender(studio("art-1"));

  expect(screen.getByRole("button", { name: "Brush a selection" })).toHaveAttribute("aria-pressed", "true");
  expect(selection()).toBe(drawn);
  expect(screen.getByRole("textbox")).toHaveValue("make the sky warmer");
});

it("writes the draft down when the page is hidden, before anything unmounts", () => {
  open(new QueryClient({ defaultOptions: { queries: { retry: false } } }));
  work();
  expect(savedStudioDraftWords("chat-studio")).toBeNull();

  Object.defineProperty(document, "visibilityState", { value: "hidden", configurable: true });
  try {
    document.dispatchEvent(new Event("visibilitychange"));
  } finally {
    delete (document as { visibilityState?: unknown }).visibilityState;
  }

  expect(savedStudioDraftWords("chat-studio")).toEqual({ instruction: "make the sky warmer", selectedId: null });
});
