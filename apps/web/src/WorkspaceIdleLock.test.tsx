/** Choosing a quiet-spell lock, and telling the server when someone uses the page. */

import { cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { useWorkspaceUse } from "./useWorkspaceUse";
import { WorkspaceLockSettings } from "./WorkspaceLockSettings";
import { resetWorkspaceLockForTests, workspaceLockView } from "./workspaceLockState";
import type { WorkspaceLockPolicy } from "./workspaceLockTypes";

vi.mock("./api", () => ({
  api: {
    workspaceLockPolicy: vi.fn(),
    updateWorkspaceLockPolicy: vi.fn(),
    workspaceLockStatus: vi.fn(),
    lockWorkspace: vi.fn(),
    noteWorkspaceUse: vi.fn(),
  },
}));

async function showWith(policy: WorkspaceLockPolicy) {
  vi.mocked(api.workspaceLockPolicy).mockResolvedValue(policy);
  render(<WorkspaceLockSettings />);
  await waitFor(() => expect(screen.getByRole("button", { name: policy.enabled ? "On" : "Off" }))
    .toHaveAttribute("aria-pressed", "true"));
}

beforeEach(() => {
  resetWorkspaceLockForTests();
  vi.mocked(api.noteWorkspaceUse).mockResolvedValue(undefined);
});

afterEach(() => {
  cleanup();
  resetWorkspaceLockForTests();
  vi.restoreAllMocks();
  vi.resetAllMocks();
});

describe("choosing a quiet spell", () => {
  it("saves a spell straight away when no PIN is set, and the window follows it", async () => {
    vi.mocked(api.updateWorkspaceLockPolicy).mockResolvedValue({ enabled: true, require_pin: false, revision: 4, idle_lock_minutes: 15 });
    vi.mocked(api.workspaceLockStatus).mockResolvedValue({
      locked: false, enabled: true, require_pin: false, lock_epoch: "neutral-epoch", idle_lock_seconds: 900,
    });
    await showWith({ enabled: true, require_pin: false, revision: 3, idle_lock_minutes: null });

    fireEvent.click(screen.getByRole("button", { name: "15 min" }));

    expect(await screen.findByRole("status")).toHaveTextContent("Lock when unused changed.");
    expect(api.updateWorkspaceLockPolicy).toHaveBeenCalledExactlyOnceWith({ expected_revision: 3, enabled: true, idle_lock_minutes: 15 });
    expect(screen.getByRole("button", { name: "15 min" })).toHaveAttribute("aria-pressed", "true");
    await waitFor(() => expect(workspaceLockView().idleSeconds).toBe(900));
  });

  it("with a PIN, a shorter spell needs nothing but a longer one asks for the PIN", async () => {
    vi.mocked(api.updateWorkspaceLockPolicy)
      .mockResolvedValueOnce({ enabled: true, require_pin: true, revision: 6, idle_lock_minutes: 5 })
      .mockResolvedValueOnce({ enabled: true, require_pin: true, revision: 7, idle_lock_minutes: null });
    vi.mocked(api.workspaceLockStatus).mockRejectedValue(new Error("neutral"));
    await showWith({ enabled: true, require_pin: true, revision: 5, idle_lock_minutes: 15 });

    fireEvent.click(screen.getByRole("button", { name: "5 min" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "5 min" })).toHaveAttribute("aria-pressed", "true"));
    expect(api.updateWorkspaceLockPolicy).toHaveBeenLastCalledWith({ expected_revision: 5, enabled: true, idle_lock_minutes: 5 });

    fireEvent.click(screen.getByRole("button", { name: "Never" }));
    expect(api.updateWorkspaceLockPolicy).toHaveBeenCalledTimes(1);
    const form = await screen.findByRole("form", { name: "Lock later when unused" });
    expect(form).toBeVisible();
    fireEvent.change(screen.getByLabelText("Current PIN"), { target: { value: "4826" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Never" })).toHaveAttribute("aria-pressed", "true"));
    expect(api.updateWorkspaceLockPolicy).toHaveBeenLastCalledWith({
      expected_revision: 6, enabled: true, current_pin: "4826", idle_lock_minutes: null,
    });
  });

  it("offers a spell only while the lock is on", async () => {
    await showWith({ enabled: false, require_pin: false, revision: 1, idle_lock_minutes: null });

    fireEvent.click(screen.getByRole("button", { name: "30 min" }));

    expect(screen.getByRole("button", { name: "30 min" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Turn the lock on to choose this.")).toBeVisible();
    expect(api.updateWorkspaceLockPolicy).not.toHaveBeenCalled();
  });
});

describe("reporting use", () => {
  it("reports a key, a click or a touch, at most once per gap, and never a pointer move", () => {
    let now = 1_000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    renderHook(() => useWorkspaceUse(300));

    fireEvent.pointerMove(window);
    expect(api.noteWorkspaceUse).not.toHaveBeenCalled();

    fireEvent.keyDown(window, { key: "a" });
    fireEvent.pointerDown(window);
    expect(api.noteWorkspaceUse).toHaveBeenCalledTimes(1);

    // A sixth of five minutes is fifty seconds; the half-minute ceiling is shorter.
    now += 29_999;
    fireEvent.touchStart(window);
    expect(api.noteWorkspaceUse).toHaveBeenCalledTimes(1);
    now += 1;
    fireEvent.touchStart(window);
    expect(api.noteWorkspaceUse).toHaveBeenCalledTimes(2);
  });

  it("reports nothing without a spell, and stops once the spell is gone", () => {
    const { rerender } = renderHook(({ idle }: { idle: number | null }) => useWorkspaceUse(idle), { initialProps: { idle: null as number | null } });

    fireEvent.keyDown(window, { key: "a" });
    expect(api.noteWorkspaceUse).not.toHaveBeenCalled();

    rerender({ idle: 60 });
    fireEvent.keyDown(window, { key: "a" });
    expect(api.noteWorkspaceUse).toHaveBeenCalledTimes(1);

    rerender({ idle: null });
    vi.spyOn(performance, "now").mockReturnValue(Number.MAX_SAFE_INTEGER);
    fireEvent.keyDown(window, { key: "a" });
    expect(api.noteWorkspaceUse).toHaveBeenCalledTimes(1);
  });
});
