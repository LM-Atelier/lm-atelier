import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { KeepAwakeSetting } from "./KeepAwakeSetting";
import { api } from "./api";
import type { KeepAwakeStatus } from "./types";

vi.mock("./api", () => ({ api: { keepAwake: vi.fn(), updateKeepAwake: vi.fn() } }));
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.resetAllMocks();
});

const status = (values: Partial<KeepAwakeStatus> = {}): KeepAwakeStatus => ({
  enabled: true,
  supported: true,
  active: false,
  running_jobs: 0,
  ...values,
});

const choices = () => screen.getByRole("group", { name: "While work runs" });

describe("keeping the computer awake while work runs", () => {
  it("shows the saved choice and what the computer is doing now", async () => {
    vi.mocked(api.keepAwake).mockResolvedValue(status({ active: true, running_jobs: 2 }));
    render(<KeepAwakeSetting />);

    expect(await screen.findByText("Keeping the computer awake while 2 jobs run.")).toBeVisible();
    expect(within(choices()).getByRole("button", { name: "Keep it awake" })).toHaveAttribute("aria-pressed", "true");
    expect(within(choices()).getByRole("button", { name: "Let it sleep" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText(/never chat/)).toHaveTextContent("The display still turns off");
  });

  it.each([
    [{ running_jobs: 0 }, "Nothing is running, so the computer sleeps on its usual schedule."],
    [{ active: true, running_jobs: 1 }, "Keeping the computer awake while 1 job runs."],
    [{ running_jobs: 1 }, "The computer refused to stay awake. The work goes on, but can stop if the computer sleeps."],
    [{ enabled: false, running_jobs: 3 }, "The computer sleeps on its usual schedule, even while work runs."],
    [{ supported: false, running_jobs: 1 }, "This computer gives LM Atelier no way to keep it awake, so long work can stop if it sleeps."],
  ])("describes %o truthfully", async (values, text) => {
    vi.mocked(api.keepAwake).mockResolvedValue(status(values));
    render(<KeepAwakeSetting />);

    expect(await screen.findByText(text)).toBeVisible();
  });

  it("turns it off, keeping focus on the choice it made", async () => {
    vi.mocked(api.keepAwake).mockResolvedValue(status({ active: true, running_jobs: 1 }));
    vi.mocked(api.updateKeepAwake).mockResolvedValue(status({ enabled: false, running_jobs: 1 }));
    render(<KeepAwakeSetting />);
    await screen.findByText("Keeping the computer awake while 1 job runs.");
    const sleep = within(choices()).getByRole("button", { name: "Let it sleep" });
    sleep.focus();

    fireEvent.click(sleep);

    expect(await screen.findByText("The computer sleeps on its usual schedule, even while work runs.")).toBeVisible();
    expect(api.updateKeepAwake).toHaveBeenCalledExactlyOnceWith({ enabled: false });
    expect(sleep).toHaveAttribute("aria-pressed", "true");
    expect(sleep).toHaveFocus();
  });

  it("sends nothing for the choice already made or before the status is known", async () => {
    let answer: (value: KeepAwakeStatus) => void = () => undefined;
    vi.mocked(api.keepAwake).mockReturnValue(new Promise((resolve) => { answer = resolve; }));
    render(<KeepAwakeSetting />);
    const awake = within(choices()).getByRole("button", { name: "Keep it awake" });
    expect(awake).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(awake);

    await act(async () => answer(status()));
    fireEvent.click(awake);

    expect(api.updateKeepAwake).not.toHaveBeenCalled();
  });

  it("says a choice that could not be saved was not changed, without the server's words", async () => {
    vi.mocked(api.keepAwake).mockResolvedValue(status());
    vi.mocked(api.updateKeepAwake).mockRejectedValue(new Error("neutral internal error marker"));
    render(<KeepAwakeSetting />);
    await screen.findByText("Nothing is running, so the computer sleeps on its usual schedule.");

    fireEvent.click(within(choices()).getByRole("button", { name: "Let it sleep" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("The setting could not be saved, so it was not changed.");
    expect(screen.queryByText("neutral internal error marker")).toBeNull();
    expect(within(choices()).getByRole("button", { name: "Keep it awake" })).toHaveAttribute("aria-pressed", "true");
  });

  it("reads the status again while the page stays open", async () => {
    vi.useFakeTimers();
    vi.mocked(api.keepAwake)
      .mockResolvedValueOnce(status())
      .mockResolvedValueOnce(status({ active: true, running_jobs: 1 }));
    render(<KeepAwakeSetting />);
    await act(async () => { await Promise.resolve(); });
    expect(screen.getByText("Nothing is running, so the computer sleeps on its usual schedule.")).toBeVisible();

    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });

    expect(screen.getByText("Keeping the computer awake while 1 job runs.")).toBeVisible();
    expect(api.keepAwake).toHaveBeenCalledTimes(2);
  });

  it("keeps a slower read that started before a save from undoing it", async () => {
    vi.useFakeTimers();
    let lateRead: (value: KeepAwakeStatus) => void = () => undefined;
    vi.mocked(api.keepAwake)
      .mockResolvedValueOnce(status())
      .mockReturnValueOnce(new Promise((resolve) => { lateRead = resolve; }));
    vi.mocked(api.updateKeepAwake).mockResolvedValue(status({ enabled: false }));
    render(<KeepAwakeSetting />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });

    fireEvent.click(within(choices()).getByRole("button", { name: "Let it sleep" }));
    await act(async () => { await Promise.resolve(); });
    await act(async () => lateRead(status()));

    expect(within(choices()).getByRole("button", { name: "Let it sleep" })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows a slow save's answer even when a read comes due while it runs", async () => {
    vi.useFakeTimers();
    let saved: (value: KeepAwakeStatus) => void = () => undefined;
    vi.mocked(api.keepAwake).mockResolvedValue(status());
    vi.mocked(api.updateKeepAwake).mockReturnValue(new Promise((resolve) => { saved = resolve; }));
    render(<KeepAwakeSetting />);
    await act(async () => { await Promise.resolve(); });

    fireEvent.click(within(choices()).getByRole("button", { name: "Let it sleep" }));
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    await act(async () => saved(status({ enabled: false })));

    expect(within(choices()).getByRole("button", { name: "Let it sleep" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("The computer sleeps on its usual schedule, even while work runs.")).toBeVisible();
  });

  it("starts no read while a save runs, so none can answer after it", async () => {
    vi.useFakeTimers();
    let saved: (value: KeepAwakeStatus) => void = () => undefined;
    let lateRead: (value: KeepAwakeStatus) => void = () => undefined;
    vi.mocked(api.keepAwake)
      .mockResolvedValueOnce(status())
      .mockReturnValueOnce(new Promise((resolve) => { lateRead = resolve; }));
    vi.mocked(api.updateKeepAwake).mockReturnValue(new Promise((resolve) => { saved = resolve; }));
    render(<KeepAwakeSetting />);
    await act(async () => { await Promise.resolve(); });

    fireEvent.click(within(choices()).getByRole("button", { name: "Let it sleep" }));
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    await act(async () => saved(status({ enabled: false })));
    await act(async () => lateRead(status()));

    expect(api.keepAwake).toHaveBeenCalledTimes(1);
    expect(within(choices()).getByRole("button", { name: "Let it sleep" })).toHaveAttribute("aria-pressed", "true");
  });

  it("says a slow save failed even when a read comes due while it runs", async () => {
    vi.useFakeTimers();
    let refuse: (reason: Error) => void = () => undefined;
    vi.mocked(api.keepAwake).mockResolvedValue(status());
    vi.mocked(api.updateKeepAwake).mockReturnValue(new Promise((_resolve, reject) => { refuse = reject; }));
    render(<KeepAwakeSetting />);
    await act(async () => { await Promise.resolve(); });

    fireEvent.click(within(choices()).getByRole("button", { name: "Let it sleep" }));
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    await act(async () => refuse(new Error("neutral internal error marker")));

    expect(screen.getByRole("alert")).toHaveTextContent("The setting could not be saved, so it was not changed.");
    expect(within(choices()).getByRole("button", { name: "Keep it awake" })).toHaveAttribute("aria-pressed", "true");
  });

  it("goes on reading the status once a save has finished", async () => {
    vi.useFakeTimers();
    vi.mocked(api.keepAwake)
      .mockResolvedValueOnce(status())
      .mockResolvedValueOnce(status({ enabled: false, supported: false }));
    vi.mocked(api.updateKeepAwake).mockResolvedValue(status({ enabled: false }));
    render(<KeepAwakeSetting />);
    await act(async () => { await Promise.resolve(); });

    fireEvent.click(within(choices()).getByRole("button", { name: "Let it sleep" }));
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });

    expect(api.keepAwake).toHaveBeenCalledTimes(2);
    expect(screen.getByText("This computer gives LM Atelier no way to keep it awake, so long work can stop if it sleeps.")).toBeVisible();
  });
});
