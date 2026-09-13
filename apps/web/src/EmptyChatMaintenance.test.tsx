/** Reviewing empty chats and deleting a chosen few, all or none. */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { CURRENT_CHAT_STORAGE_KEY, EmptyChatMaintenance } from "./EmptyChatMaintenance";
import type { EmptyChatEntry, EmptyChatPage, EmptyChatPreview } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      emptyChats: vi.fn(),
      previewEmptyChats: vi.fn(),
      deleteEmptyChats: vi.fn(),
    },
  };
});

function entry(id: string, overrides: Partial<EmptyChatEntry> = {}): EmptyChatEntry {
  return {
    id,
    classification: "strict_blank",
    created_at: "2026-09-10T00:00:00Z",
    updated_at: "2026-09-10T00:00:00Z",
    age_hours: 72,
    reasons: [],
    deletable: true,
    ...overrides,
  };
}

function page(entries: EmptyChatEntry[]): EmptyChatPage {
  return { entries, next_cursor: null, counts: {}, evaluated_at: "2026-09-13T00:00:00Z" };
}

function preview(overrides: Partial<EmptyChatPreview> = {}): EmptyChatPreview {
  return {
    preview_id: "emptyprev_one",
    digest: "d".repeat(64),
    expires_at: "2026-09-13T00:10:00Z",
    strict_count: 1,
    configured_count: 0,
    conflicts: [],
    ...overrides,
  };
}

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <EmptyChatMaintenance />
    </QueryClientProvider>,
  );
}

function deleteButton() {
  return screen.getByRole("button", { name: /^Delete selected/ });
}

describe("EmptyChatMaintenance", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("opens with nothing selected, and deleting asks for a choice first", async () => {
    vi.mocked(api.emptyChats).mockResolvedValue(page([entry("chat_a"), entry("chat_b")]));

    show();

    await waitFor(() => expect(screen.getByText("2 empty chats, 0 selected.")).toBeTruthy());
    for (const box of screen.getAllByRole("checkbox", { name: /old$/ })) {
      expect((box as HTMLInputElement).checked).toBe(false);
    }
    expect(deleteButton().getAttribute("aria-disabled")).toBe("true");
    fireEvent.click(deleteButton());
    expect(api.previewEmptyChats).not.toHaveBeenCalled();
  });

  it("selects only untouched chats, and never the chat open in the workspace", async () => {
    localStorage.setItem(CURRENT_CHAT_STORAGE_KEY, "chat_open");
    vi.mocked(api.emptyChats).mockResolvedValue(
      page([
        entry("chat_untouched"),
        entry("chat_open"),
        entry("chat_named", { classification: "configured_blank", reasons: ["custom_title"] }),
        entry("chat_broken", {
          classification: "inconsistent",
          reasons: ["stale_head"],
          deletable: false,
        }),
      ]),
    );
    vi.mocked(api.previewEmptyChats).mockResolvedValue(preview());

    show();
    await waitFor(() => expect(screen.getByRole("button", { name: "Select untouched" })).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "Select untouched" }));
    fireEvent.click(deleteButton());

    await waitFor(() => expect(api.previewEmptyChats).toHaveBeenCalled());
    expect(api.previewEmptyChats).toHaveBeenCalledWith({
      chat_ids: ["chat_untouched"],
      include_archived: false,
      include_configured: false,
    });
  });

  it("lists a chat whose records disagree without offering it", async () => {
    vi.mocked(api.emptyChats).mockResolvedValue(
      page([entry("chat_broken", { classification: "inconsistent", reasons: ["stale_head"], deletable: false })]),
    );

    show();

    await waitFor(() => expect(screen.getByText("Not offered")).toBeTruthy());
    expect(screen.queryAllByRole("checkbox", { name: /old$/ })).toHaveLength(0);
    expect(screen.getByText(/its records disagree with each other/)).toBeTruthy();
  });

  it("confirms the server's counts and deletes only the chats the check bound", async () => {
    vi.mocked(api.emptyChats).mockResolvedValue(page([entry("chat_a"), entry("chat_b")]));
    vi.mocked(api.previewEmptyChats).mockResolvedValue(
      preview({ strict_count: 1, conflicts: [{ chat_id: "chat_b", reason: "not_empty" }] }),
    );
    vi.mocked(api.deleteEmptyChats).mockResolvedValue({
      operation_id: "op",
      deleted_ids: ["chat_a"],
      deleted_at: "2026-09-13T00:01:00Z",
      replayed: false,
    });

    show();
    await waitFor(() => expect(screen.getAllByRole("checkbox", { name: /old$/ })).toHaveLength(2));
    for (const box of screen.getAllByRole("checkbox", { name: /old$/ })) fireEvent.click(box);
    fireEvent.click(deleteButton());

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("1 chat, none of which you set up.")).toBeTruthy();
    expect(within(dialog).getByText(/no longer empty/)).toBeTruthy();
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(api.deleteEmptyChats).toHaveBeenCalled());
    const body = vi.mocked(api.deleteEmptyChats).mock.calls[0][0];
    expect(body).toMatchObject({
      preview_id: "emptyprev_one",
      digest: "d".repeat(64),
      acknowledged_count: 1,
      acknowledged_configured: false,
      chat_ids: ["chat_a"],
      include_archived: false,
      include_configured: false,
    });
    expect(body.operation_id).toMatch(/\S/);
    await waitFor(() => expect(screen.getByText("Deleted 1 empty chat.")).toBeTruthy());
  });

  it("will not delete chats somebody set up without their own confirmation", async () => {
    vi.mocked(api.emptyChats).mockResolvedValue(
      page([entry("chat_named", { classification: "configured_blank", reasons: ["pinned"] })]),
    );
    vi.mocked(api.previewEmptyChats).mockResolvedValue(
      preview({ strict_count: 0, configured_count: 1 }),
    );
    vi.mocked(api.deleteEmptyChats).mockResolvedValue({
      operation_id: "op",
      deleted_ids: ["chat_named"],
      deleted_at: "2026-09-13T00:01:00Z",
      replayed: false,
    });

    show();
    fireEvent.click(await screen.findByRole("checkbox", { name: "Include chats you set up" }));
    const box = await screen.findByRole("checkbox", { name: /^Set up but unused/ });
    fireEvent.click(box);
    fireEvent.click(deleteButton());

    const dialog = await screen.findByRole("dialog");
    const confirm = within(dialog).getByRole("button", { name: "Delete" });
    fireEvent.click(confirm);
    expect(api.deleteEmptyChats).not.toHaveBeenCalled();

    fireEvent.click(within(dialog).getByRole("checkbox", { name: "Also delete the chats I set up" }));
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(api.deleteEmptyChats).toHaveBeenCalled());
    expect(vi.mocked(api.deleteEmptyChats).mock.calls[0][0]).toMatchObject({
      acknowledged_configured: true,
      include_configured: true,
    });
  });

  it("says what to do next when the server refuses, and reports no deletion", async () => {
    vi.mocked(api.emptyChats).mockResolvedValue(page([entry("chat_a")]));
    vi.mocked(api.previewEmptyChats).mockResolvedValue(preview());
    vi.mocked(api.deleteEmptyChats).mockRejectedValue(
      new ApiError(409, {}, "refused", "empty-chat-selection-drifted"),
    );

    show();
    fireEvent.click(await screen.findByRole("checkbox", { name: /old$/ }));
    fireEvent.click(deleteButton());
    fireEvent.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Delete" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("changed while you were deciding");
    expect(screen.queryByText(/^Deleted/)).toBeNull();
  });

  it("reads the open chat under the key the workspace writes it", () => {
    // The key is duplicated rather than imported, because the workspace keeps it
    // private; this is what stops the two from drifting apart silently.
    const app = readFileSync(join(__dirname, "App.tsx"), "utf8");
    expect(app).toContain(`const CURRENT_CHAT_KEY = "${CURRENT_CHAT_STORAGE_KEY}";`);
  });
});
