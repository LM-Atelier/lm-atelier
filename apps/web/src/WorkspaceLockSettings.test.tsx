import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { WorkspaceLockSettings } from "./WorkspaceLockSettings";
import { resetWorkspaceLockForTests, workspaceLockView } from "./workspaceLockState";

vi.mock("./api", () => ({
  api: { workspaceLockPolicy: vi.fn(), updateWorkspaceLockPolicy: vi.fn(), lockWorkspace: vi.fn() },
}));

function refusal(status: number, code: string, extra: Record<string, unknown> = {}): Error {
  return Object.assign(new Error("neutral refusal"), { status, code, payload: { code, ...extra } });
}

async function showWith(policy: { enabled: boolean; require_pin: boolean; revision: number }) {
  vi.mocked(api.workspaceLockPolicy).mockResolvedValue(policy);
  render(<WorkspaceLockSettings />);
  await waitFor(() => expect(screen.getByRole("button", { name: policy.enabled ? "On" : "Off" }))
    .toHaveAttribute("aria-pressed", "true"));
}

function type(label: string, value: string) {
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
}

beforeEach(() => { resetWorkspaceLockForTests(); });

afterEach(() => {
  cleanup();
  resetWorkspaceLockForTests();
  vi.resetAllMocks();
});

describe("workspace lock settings", () => {
  it("turns the lock on against the revision it read", async () => {
    vi.mocked(api.updateWorkspaceLockPolicy).mockResolvedValue({ enabled: true, require_pin: false, revision: 3 });
    await showWith({ enabled: false, require_pin: false, revision: 2 });

    fireEvent.click(screen.getByRole("button", { name: "On" }));

    expect(await screen.findByRole("status")).toHaveTextContent("Workspace lock turned on.");
    expect(api.updateWorkspaceLockPolicy).toHaveBeenCalledExactlyOnceWith({ expected_revision: 2, enabled: true });
    expect(screen.getByRole("button", { name: "On" })).toHaveAttribute("aria-pressed", "true");
  });

  it("reloads the setting when another window changed it first, and asks for a second look", async () => {
    vi.mocked(api.workspaceLockPolicy)
      .mockResolvedValueOnce({ enabled: false, require_pin: false, revision: 2 })
      .mockResolvedValueOnce({ enabled: true, require_pin: true, revision: 4 });
    vi.mocked(api.updateWorkspaceLockPolicy).mockRejectedValue(refusal(409, "workspace-lock-setting-stale", { current_revision: 4 }));
    render(<WorkspaceLockSettings />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Off" })).toHaveAttribute("aria-pressed", "true"));

    fireEvent.click(screen.getByRole("button", { name: "On" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("The lock setting was changed somewhere else. Check it below and try again.");
    await waitFor(() => expect(screen.getByRole("button", { name: "On" })).toHaveAttribute("aria-pressed", "true"));
    expect(screen.getByText("Unlocking asks for the PIN.")).toBeVisible();
    expect(api.workspaceLockPolicy).toHaveBeenCalledTimes(2);
    expect(api.updateWorkspaceLockPolicy).toHaveBeenCalledTimes(1);
  });

  it("asks for the current PIN before turning the lock off, and sends it only in the change", async () => {
    vi.mocked(api.updateWorkspaceLockPolicy).mockResolvedValue({ enabled: false, require_pin: true, revision: 6 });
    await showWith({ enabled: true, require_pin: true, revision: 5 });

    fireEvent.click(screen.getByRole("button", { name: "Off" }));
    const form = screen.getByRole("form", { name: "Turn the lock off" });
    expect(screen.getByLabelText("Current PIN")).toHaveFocus();
    fireEvent.submit(form);
    expect(screen.getByRole("alert")).toHaveTextContent("Enter the current PIN.");
    expect(api.updateWorkspaceLockPolicy).not.toHaveBeenCalled();

    type("Current PIN", "1234");
    fireEvent.submit(form);

    expect(await screen.findByRole("status")).toHaveTextContent("Workspace lock turned off.");
    expect(api.updateWorkspaceLockPolicy).toHaveBeenCalledExactlyOnceWith({ expected_revision: 5, enabled: false, current_pin: "1234" });
  });

  it("asks for the current PIN to change it, and checks the new one twice", async () => {
    vi.mocked(api.updateWorkspaceLockPolicy).mockResolvedValue({ enabled: true, require_pin: true, revision: 8 });
    await showWith({ enabled: true, require_pin: true, revision: 7 });

    fireEvent.click(screen.getByRole("button", { name: "Change PIN" }));
    const form = screen.getByRole("form", { name: "Change the PIN" });
    type("Current PIN", "1234");
    type("New PIN", "5678");
    type("Confirm new PIN", "5679");
    fireEvent.submit(form);
    expect(screen.getByRole("alert")).toHaveTextContent("The two PINs do not match.");
    for (const label of ["Current PIN", "New PIN", "Confirm new PIN"]) expect(screen.getByLabelText(label)).toHaveValue("");

    type("Current PIN", "1234");
    type("New PIN", "567");
    type("Confirm new PIN", "567");
    fireEvent.submit(form);
    expect(screen.getByRole("alert")).toHaveTextContent("A PIN needs 4 to 64 characters.");
    expect(api.updateWorkspaceLockPolicy).not.toHaveBeenCalled();

    type("Current PIN", "1234");
    type("New PIN", "5678");
    type("Confirm new PIN", "5678");
    fireEvent.submit(form);

    expect(await screen.findByRole("status")).toHaveTextContent("PIN changed.");
    expect(api.updateWorkspaceLockPolicy).toHaveBeenCalledExactlyOnceWith({
      expected_revision: 7, enabled: true, current_pin: "1234", new_pin: "5678",
    });
  });

  it("asks for the current PIN to remove it, and empties the field when it is refused", async () => {
    vi.mocked(api.updateWorkspaceLockPolicy).mockRejectedValue(refusal(403, "workspace-pin-refused", { retry_after_seconds: 0 }));
    await showWith({ enabled: true, require_pin: true, revision: 9 });

    fireEvent.click(screen.getByRole("button", { name: "Remove PIN" }));
    type("Current PIN", "4321");
    fireEvent.submit(screen.getByRole("form", { name: "Remove the PIN" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("The current PIN was not accepted.");
    expect(api.updateWorkspaceLockPolicy).toHaveBeenCalledExactlyOnceWith({
      expected_revision: 9, enabled: true, current_pin: "4321", clear_pin: true,
    });
    expect(screen.getByLabelText("Current PIN")).toHaveValue("");
    expect(document.body.textContent).not.toContain("4321");
  });

  it("sets a first PIN without asking for a current one", async () => {
    vi.mocked(api.updateWorkspaceLockPolicy).mockResolvedValue({ enabled: true, require_pin: true, revision: 2 });
    await showWith({ enabled: true, require_pin: false, revision: 1 });

    fireEvent.click(screen.getByRole("button", { name: "Set PIN" }));
    expect(screen.queryByLabelText("Current PIN")).toBeNull();
    expect(screen.getByLabelText("New PIN")).toHaveFocus();
    type("New PIN", "2580");
    type("Confirm new PIN", "2580");
    fireEvent.submit(screen.getByRole("form", { name: "Set a PIN" }));

    expect(await screen.findByRole("status")).toHaveTextContent("PIN saved.");
    expect(api.updateWorkspaceLockPolicy).toHaveBeenCalledExactlyOnceWith({ expected_revision: 1, enabled: true, new_pin: "2580" });
    expect(screen.getByRole("button", { name: "Change PIN" })).toBeVisible();
  });

  it("locks this window at once with the status the server returns", async () => {
    vi.mocked(api.lockWorkspace).mockResolvedValue({ locked: true, enabled: true, require_pin: false, lock_epoch: "epoch-one" });
    await showWith({ enabled: true, require_pin: false, revision: 1 });

    fireEvent.click(screen.getByRole("button", { name: "Lock now" }));

    await waitFor(() => expect(workspaceLockView().phase).toBe("locked"));
    expect(workspaceLockView().epoch).toBe("epoch-one");
  });

  it("offers Lock now only while the lock is on", async () => {
    await showWith({ enabled: false, require_pin: false, revision: 0 });

    const lockNow = screen.getByRole("button", { name: "Lock now" });
    expect(lockNow).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(lockNow);

    expect(api.lockWorkspace).not.toHaveBeenCalled();
    expect(screen.getByText("Turn the lock on to use this.")).toBeVisible();
  });
});
