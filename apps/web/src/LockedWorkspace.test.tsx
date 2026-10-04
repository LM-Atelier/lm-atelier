import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { LockedWorkspace } from "./LockedWorkspace";
import { noteWorkspaceLocked, resetWorkspaceLockForTests, workspaceLockView } from "./workspaceLockState";

vi.mock("./api", () => ({ api: { unlockWorkspace: vi.fn() } }));

function refusal(status: number, code: string, retryAfterSeconds = 0): Error {
  return Object.assign(new Error("neutral refusal"), { status, code, payload: { code, retry_after_seconds: retryAfterSeconds } });
}

function pinField(): HTMLInputElement {
  return screen.getByLabelText("PIN") as HTMLInputElement;
}

function unlockButton(): HTMLElement {
  return screen.getByRole("button", { name: /Unlock/ });
}

async function tryPin(value: string) {
  fireEvent.change(pinField(), { target: { value } });
  await act(async () => { fireEvent.click(unlockButton()); });
}

beforeEach(() => {
  resetWorkspaceLockForTests();
  noteWorkspaceLocked();
});

afterEach(() => {
  cleanup();
  resetWorkspaceLockForTests();
  vi.useRealTimers();
  vi.resetAllMocks();
});

describe("the locked workspace", () => {
  it("asks for the PIN in a labelled password field that already has focus", () => {
    render(<LockedWorkspace requirePin />);

    expect(screen.getByRole("heading", { name: "LM Atelier is locked" })).toBeVisible();
    expect(pinField()).toHaveAttribute("type", "password");
    expect(pinField()).toHaveAttribute("autocomplete", "off");
    expect(pinField()).toHaveFocus();
    expect(screen.getByRole("main")).toHaveAttribute("id", "main-content");
  });

  it("empties the field after every attempt and sends the PIN only to the unlock request", async () => {
    vi.mocked(api.unlockWorkspace).mockRejectedValue(refusal(403, "workspace-pin-refused"));
    render(<LockedWorkspace requirePin />);

    await tryPin("2468");

    expect(pinField()).toHaveValue("");
    await screen.findByRole("alert");
    expect(api.unlockWorkspace).toHaveBeenCalledExactlyOnceWith("2468");
    expect(pinField()).toHaveAttribute("aria-invalid", "true");
    expect(pinField()).toHaveFocus();
  });

  it("says the same thing however an attempt fails, and never repeats what was typed", async () => {
    render(<LockedWorkspace requirePin />);
    const messages: string[] = [];
    for (const failure of [
      refusal(403, "workspace-pin-refused"),
      refusal(422, "request-validation-invalid"),
      refusal(503, "workspace-pin-unavailable"),
      new Error("neutral network failure"),
    ]) {
      vi.mocked(api.unlockWorkspace).mockRejectedValueOnce(failure);
      await tryPin("1357");
      messages.push((await screen.findByRole("alert")).textContent ?? "");
    }

    expect(new Set(messages)).toEqual(new Set(["LM Atelier could not be unlocked."]));
    expect(api.unlockWorkspace).toHaveBeenCalledTimes(4);
    expect(document.body.textContent).not.toContain("1357");
    expect(document.body.textContent).not.toMatch(/attempts? (left|remaining)/i);
  });

  it("does not send an empty PIN, which the server would count as a failed attempt", async () => {
    render(<LockedWorkspace requirePin />);

    await act(async () => { fireEvent.click(unlockButton()); });

    expect(api.unlockWorkspace).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("Enter the PIN.");
  });

  it("waits out a throttled attempt with the button unavailable, then offers it again", async () => {
    vi.useFakeTimers();
    vi.mocked(api.unlockWorkspace).mockRejectedValueOnce(refusal(429, "workspace-pin-throttled", 2));
    render(<LockedWorkspace requirePin />);
    const settle = (milliseconds: number) => act(async () => { await vi.advanceTimersByTimeAsync(milliseconds); });

    await tryPin("9753");
    await settle(0);

    expect(unlockButton()).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Try again in 2 seconds.")).toBeVisible();
    await tryPin("9753");
    await settle(0);
    expect(api.unlockWorkspace).toHaveBeenCalledTimes(1);

    await settle(1_000);
    expect(screen.getByText("Try again in 1 second.")).toBeVisible();
    await settle(1_000);

    expect(unlockButton()).toHaveAttribute("aria-disabled", "false");
    expect(screen.queryByText(/Try again in/)).toBeNull();
  });

  it("explains a lock setting that cannot be read, since no PIN will open it", async () => {
    vi.mocked(api.unlockWorkspace).mockRejectedValueOnce(refusal(409, "workspace-lock-setting-invalid"));
    render(<LockedWorkspace requirePin />);

    await tryPin("8642");

    expect(await screen.findByRole("alert")).toHaveTextContent("The saved lock setting could not be read");
  });

  it("offers a single focused Unlock button when no PIN is set, and unlocks with it", async () => {
    vi.mocked(api.unlockWorkspace).mockResolvedValue({ locked: false, enabled: true, require_pin: false, lock_epoch: "epoch-one" });
    render(<LockedWorkspace requirePin={false} />);

    expect(screen.queryByLabelText("PIN")).toBeNull();
    expect(unlockButton()).toHaveFocus();
    await act(async () => { fireEvent.click(unlockButton()); });

    await waitFor(() => expect(workspaceLockView().phase).toBe("unlocked"));
    expect(api.unlockWorkspace).toHaveBeenCalledExactlyOnceWith(undefined);
    expect(workspaceLockView().epoch).toBe("epoch-one");
  });
});
