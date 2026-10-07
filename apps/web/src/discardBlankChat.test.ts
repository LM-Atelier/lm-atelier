import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { discardBlankChat } from "./discardBlankChat";

vi.mock("./api", () => ({ api: { previewEmptyChats: vi.fn(), deleteEmptyChats: vi.fn() } }));
afterEach(() => { vi.resetAllMocks(); });

const choice = { chat_ids: ["chat_new"], min_age_hours: 0, include_archived: false, include_configured: false };

function preview(overrides: Record<string, unknown> = {}) {
  return {
    preview_id: "preview_1",
    digest: "d".repeat(64),
    expires_at: "2026-10-03T00:00:00Z",
    strict_count: 1,
    configured_count: 0,
    conflicts: [],
    ...overrides,
  };
}

describe("removing a chat a refused start left behind", () => {
  it("removes it as an empty chat, of any age, when it is strictly blank", async () => {
    vi.mocked(api.previewEmptyChats).mockResolvedValue(preview());
    vi.mocked(api.deleteEmptyChats).mockResolvedValue(undefined as never);

    await discardBlankChat("chat_new");

    expect(api.previewEmptyChats).toHaveBeenCalledExactlyOnceWith(choice);
    expect(api.deleteEmptyChats).toHaveBeenCalledExactlyOnceWith({
      ...choice,
      operation_id: expect.any(String),
      preview_id: "preview_1",
      digest: "d".repeat(64),
      acknowledged_count: 1,
      acknowledged_configured: false,
    });
  });

  it.each([
    ["holds something", { strict_count: 0, conflicts: [{ chat_id: "chat_new", reason: "not_empty" }] }],
    ["is set apart by its settings", { strict_count: 0, configured_count: 1 }],
    ["is not there", { strict_count: 0, conflicts: [{ chat_id: "chat_new", reason: "missing" }] }],
  ])("leaves a chat that %s", async (_case, overrides) => {
    vi.mocked(api.previewEmptyChats).mockResolvedValue(preview(overrides));

    await discardBlankChat("chat_new");

    expect(api.deleteEmptyChats).not.toHaveBeenCalled();
  });

  it("never fails, leaving the chat when the cleanup cannot be asked", async () => {
    vi.mocked(api.previewEmptyChats).mockRejectedValue(new Error("offline"));

    await expect(discardBlankChat("chat_new")).resolves.toBeUndefined();
    expect(api.deleteEmptyChats).not.toHaveBeenCalled();
  });
});
