/** Withdrawing a backup restore that waits for the next start. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ScheduledRestoreNotice } from "./ScheduledRestoreNotice";

vi.mock("./api", () => ({ api: { cancelRestore: vi.fn() } }));

function show(onCancelled = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  client.setQueryData(["backups"], []);
  client.setQueryData(["backups", "restore-state"], { state: "pending" });
  render(<QueryClientProvider client={client}><ScheduledRestoreNotice onCancelled={onCancelled} /></QueryClientProvider>);
  return { client, onCancelled };
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("cancels the scheduled restore and refreshes the backups", async () => {
  vi.mocked(api.cancelRestore).mockResolvedValue(undefined);
  const { client, onCancelled } = show();

  expect(screen.getByRole("status")).toHaveTextContent("Restore scheduled. Restart LM Atelier to apply the selected backup.");
  fireEvent.click(screen.getByRole("button", { name: "Cancel restore" }));

  await waitFor(() => expect(onCancelled).toHaveBeenCalledTimes(1));
  expect(api.cancelRestore).toHaveBeenCalledTimes(1);
  await waitFor(() => expect(client.getQueryState(["backups"])?.isInvalidated).toBe(true));
  expect(client.getQueryState(["backups", "restore-state"])?.isInvalidated).toBe(true);
});

it("says so when the restore cannot be cancelled, and keeps the notice", async () => {
  vi.mocked(api.cancelRestore).mockRejectedValue(new Error("refused"));
  const { client, onCancelled } = show();

  fireEvent.click(screen.getByRole("button", { name: "Cancel restore" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("The restore could not be cancelled. Try again.");
  expect(screen.getByRole("status")).toHaveTextContent("Restore scheduled.");
  expect(screen.getByRole("button", { name: "Cancel restore" })).toBeInTheDocument();
  expect(onCancelled).not.toHaveBeenCalled();
  expect(client.getQueryState(["backups"])?.isInvalidated).toBe(false);
});

it("sends one cancel while the first is still on its way", async () => {
  let finish: () => void = () => undefined;
  vi.mocked(api.cancelRestore).mockReturnValue(new Promise<void>((resolve) => { finish = resolve; }));
  const { onCancelled } = show();

  fireEvent.click(screen.getByRole("button", { name: "Cancel restore" }));
  const pending = await screen.findByRole("button", { name: "Cancelling…" });
  expect(pending).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(pending);
  expect(api.cancelRestore).toHaveBeenCalledTimes(1);

  finish();
  await waitFor(() => expect(onCancelled).toHaveBeenCalledTimes(1));
});
