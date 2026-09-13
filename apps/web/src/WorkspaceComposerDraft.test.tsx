/** A chat's composer draft kept in the workspace: read into an empty composer, saved as typing pauses. */

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect, useState } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { DRAFT_SAVE_DELAY_MS, composerDraftFrom, storedDraft } from "./composerDraftStore";
import { EMPTY_COMPOSER_DRAFT, type ComposerDraft, type ComposerDraftUpdate } from "./composerPromptSource";
import type { ChatComposerDraft } from "./types";
import { initialTurnEditorState } from "./useTurnEditorState";
import { WorkspaceComposerDraft } from "./WorkspaceComposerDraft";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { composerDraft: vi.fn(), saveComposerDraft: vi.fn() } };
});

const STORED: ChatComposerDraft = {
  chat_id: "chat-garden",
  revision: 4,
  updated_at: "2026-09-01T00:00:00Z",
  text: "A wider path between the beds",
  prompt_source: null,
  mode: "image",
  output_count: 3,
  attachments: [{ artifact_id: "sha256:sketch", kind: "image", origin: "uploaded" }],
  mentions: [{ reference_subject_id: "refsub-garden", mention_slug: "garden" }],
  template_settings: null,
};

const NONE: ChatComposerDraft = {
  ...STORED, revision: 0, text: "", mode: "auto", output_count: 1, attachments: [], mentions: [],
};

function typed(text: string): ComposerDraft {
  return { text, promptSource: null, editor: initialTurnEditorState("auto", undefined) };
}

/** What the composer shows, and how the test types into it. */
const probe: { shown: ComposerDraft; type: (draft: ComposerDraft) => void } = {
  shown: EMPTY_COMPOSER_DRAFT,
  type: () => undefined,
};

/** The composer as the workspace holds it: an update re-renders with the new draft. */
function Composer({ chatId = "chat-garden", rerenders = true }: { chatId?: string; rerenders?: boolean }) {
  const [draft, setDraft] = useState<ComposerDraft>(EMPTY_COMPOSER_DRAFT);
  useEffect(() => {
    probe.shown = draft;
    probe.type = setDraft;
  }, [draft]);
  const onDraft = (update: ComposerDraftUpdate) => {
    if (!rerenders) {
      probe.shown = typeof update === "function" ? update(probe.shown) : update;
      return;
    }
    setDraft((current) => (typeof update === "function" ? update(current) : update));
  };
  return <WorkspaceComposerDraft chatId={chatId} draft={draft} onDraft={onDraft} />;
}

async function settle() {
  await act(async () => {
    for (let turn = 0; turn < 5; turn += 1) await Promise.resolve();
  });
}

async function pause() {
  await act(async () => {
    vi.advanceTimersByTime(DRAFT_SAVE_DELAY_MS);
  });
  await settle();
}

async function show(draft: ComposerDraft) {
  await act(async () => probe.type(draft));
}

function savedRevisions(): number[] {
  return vi.mocked(api.saveComposerDraft).mock.calls.map(([, expected]) => expected);
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(api.saveComposerDraft).mockImplementation(async (chatId, expected, draft) => ({
    ...STORED, ...draft, chat_id: chatId, revision: expected + 1,
  }));
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.clearAllMocks();
});

it("puts a stored draft into an empty composer, attachments and all", async () => {
  vi.mocked(api.composerDraft).mockResolvedValue(STORED);
  render(<Composer />);
  await settle();

  expect(probe.shown).toMatchObject({
    text: "A wider path between the beds",
    editor: {
      mode: "image",
      outputCount: 3,
      attachments: [{ id: "sha256:sketch", kind: "image", origin: "uploaded" }],
      mentions: [{ referenceSubjectId: "refsub-garden", mentionSlug: "garden" }],
    },
  });
});

it("never replaces what was typed before the stored draft arrived, and saves nothing before it does", async () => {
  let arrive: (draft: ChatComposerDraft) => void = () => undefined;
  vi.mocked(api.composerDraft).mockReturnValue(new Promise((resolve) => { arrive = resolve; }));
  render(<Composer />);
  await pause();
  await show(typed("Typed while it loaded"));
  await pause();
  expect(api.saveComposerDraft).not.toHaveBeenCalled();

  await act(async () => arrive(STORED));
  await settle();

  expect(probe.shown.text).toBe("Typed while it loaded");
});

it("saves once typing pauses, naming the revision it read, and not before", async () => {
  vi.mocked(api.composerDraft).mockResolvedValue(STORED);
  render(<Composer />);
  await settle();
  // Reading the stored draft is not a change, so nothing is saved for it.
  await pause();
  expect(api.saveComposerDraft).not.toHaveBeenCalled();

  await show({ ...probe.shown, text: "A wider path, and a bench" });
  await act(async () => {
    vi.advanceTimersByTime(DRAFT_SAVE_DELAY_MS - 1);
  });
  expect(api.saveComposerDraft).not.toHaveBeenCalled();
  await pause();

  expect(savedRevisions()).toEqual([4]);
  expect(vi.mocked(api.saveComposerDraft).mock.calls[0][2]).toMatchObject({
    text: "A wider path, and a bench",
    mode: "image",
    output_count: 3,
    attachments: [{ artifact_id: "sha256:sketch", kind: "image", origin: "uploaded" }],
  });
});

it("keeps a stored draft it just read even if a save comes due before the composer shows it", async () => {
  vi.mocked(api.composerDraft).mockResolvedValue(STORED);
  render(<Composer rerenders={false} />);
  await settle();

  await pause();

  expect(api.saveComposerDraft).not.toHaveBeenCalled();
  expect(probe.shown.text).toBe("A wider path between the beds");
});

it("saves in order, so a save still on its way cannot bring back a sent draft", async () => {
  vi.mocked(api.composerDraft).mockResolvedValue(NONE);
  let finishFirst: () => void = () => undefined;
  vi.mocked(api.saveComposerDraft).mockImplementationOnce(
    (chatId, expected, draft) => new Promise((resolve) => {
      finishFirst = () => resolve({ ...STORED, ...draft, chat_id: chatId, revision: expected + 1 });
    }),
  );
  render(<Composer />);
  await settle();

  await show(typed("Plant the tulips first"));
  await pause();
  // Sent while that save is still on its way: the composer empties.
  await show(EMPTY_COMPOSER_DRAFT);
  await pause();
  expect(savedRevisions()).toEqual([0]);

  await act(async () => finishFirst());
  await settle();

  expect(savedRevisions()).toEqual([0, 1]);
  expect(vi.mocked(api.saveComposerDraft).mock.calls[1][2]).toMatchObject({ text: "", attachments: [] });
});

it("never overwrites another window's draft, and lets the person choose", async () => {
  const other = { ...STORED, revision: 7, text: "From the other window" };
  vi.mocked(api.composerDraft).mockResolvedValueOnce(NONE).mockResolvedValue(other);
  vi.mocked(api.saveComposerDraft).mockRejectedValueOnce(
    new ApiError(409, "stale", "stale", "chat-draft-revision-stale"),
  );
  render(<Composer />);
  await settle();

  await show(typed("This window's words"));
  await pause();

  expect(screen.getByRole("alert")).toHaveTextContent("also changed in another window");
  expect(probe.shown.text).toBe("This window's words");
  await show(typed("This window's words, and more"));
  await pause();
  expect(savedRevisions()).toEqual([0]);

  fireEvent.click(screen.getByRole("button", { name: "Keep this draft" }));
  await settle();
  expect(savedRevisions()).toEqual([0, 7]);
  expect(vi.mocked(api.saveComposerDraft).mock.calls[1][2].text).toBe("This window's words, and more");
  expect(screen.queryByRole("alert")).toBeNull();
});

it("takes the other window's draft only when asked", async () => {
  const other = { ...STORED, revision: 7, text: "From the other window" };
  vi.mocked(api.composerDraft).mockResolvedValueOnce(NONE).mockResolvedValue(other);
  vi.mocked(api.saveComposerDraft).mockRejectedValueOnce(
    new ApiError(409, "stale", "stale", "chat-draft-revision-stale"),
  );
  render(<Composer />);
  await settle();
  await show(typed("This window's words"));
  await pause();

  fireEvent.click(screen.getByRole("button", { name: "Use the other one" }));
  await settle();
  await pause();

  expect(probe.shown.text).toBe("From the other window");
  expect(savedRevisions()).toEqual([0]);
});

it("saves an emptied draft as empty, and asks nothing when there never was one", async () => {
  vi.mocked(api.composerDraft).mockResolvedValue(STORED);
  render(<Composer />);
  await settle();

  await show(EMPTY_COMPOSER_DRAFT);
  await pause();
  expect(savedRevisions()).toEqual([4]);
  expect(vi.mocked(api.saveComposerDraft).mock.calls[0][2]).toMatchObject({ text: "", attachments: [] });

  cleanup();
  vi.mocked(api.saveComposerDraft).mockClear();
  vi.mocked(api.composerDraft).mockResolvedValue(NONE);
  render(<Composer chatId="chat-empty" />);
  await settle();
  await show(typed(""));
  await pause();
  // Choosing a mode alone is not a draft worth creating.
  await show({ ...typed(""), editor: initialTurnEditorState("image", undefined) });
  await pause();
  expect(api.saveComposerDraft).not.toHaveBeenCalled();
});

it("tries to save what is pending when the chat is left", async () => {
  vi.mocked(api.composerDraft).mockResolvedValue(NONE);
  const view = render(<Composer />);
  await settle();

  await show(typed("Half a thought"));
  view.unmount();
  await settle();

  expect(vi.mocked(api.saveComposerDraft).mock.calls.map(([, , draft]) => draft.text)).toEqual(["Half a thought"]);
});

it("stores and restores the same draft", () => {
  expect(storedDraft(composerDraftFrom(STORED))).toEqual({
    text: STORED.text,
    prompt_source: null,
    mode: "image",
    output_count: 3,
    attachments: STORED.attachments,
    mentions: STORED.mentions,
    template_settings: null,
  });
});
