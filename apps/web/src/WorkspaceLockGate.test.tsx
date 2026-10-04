import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { WorkspaceLockGate } from "./WorkspaceLockGate";
import { noteWorkspaceLocked, resetWorkspaceLockForTests } from "./workspaceLockState";

vi.mock("./api", () => ({ api: { workspaceLockStatus: vi.fn(), unlockWorkspace: vi.fn() } }));

const unlocked = { locked: false, enabled: true, require_pin: false, lock_epoch: "epoch-one" };
const lockedWithoutPin = { locked: true, enabled: true, require_pin: false, lock_epoch: "epoch-two" };
const lockedWithPin = { ...lockedWithoutPin, require_pin: true };

let mounts = 0;
function Workspace() {
  const notes = useQuery({ queryKey: ["notes"], queryFn: async () => "Pottery notes" });
  return <p>Workspace content {notes.data ?? ""}</p>;
}
function CountedWorkspace() {
  useEffect(() => { mounts += 1; }, []);
  return <Workspace />;
}

const clients: QueryClient[] = [];
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(
    <QueryClientProvider client={client}>
      <WorkspaceLockGate><CountedWorkspace /></WorkspaceLockGate>
    </QueryClientProvider>,
  );
  return client;
}

beforeEach(() => {
  mounts = 0;
  resetWorkspaceLockForTests();
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  resetWorkspaceLockForTests();
  vi.useRealTimers();
  vi.resetAllMocks();
});

describe("the workspace lock gate", () => {
  it("shows a neutral page until it knows whether the workspace is locked", () => {
    vi.mocked(api.workspaceLockStatus).mockReturnValue(new Promise(() => undefined));
    show();

    expect(screen.getByRole("status")).toHaveTextContent("Opening LM Atelier…");
    expect(screen.queryByText(/Workspace content/)).toBeNull();
  });

  it("shows the locked page in place of the workspace", async () => {
    vi.mocked(api.workspaceLockStatus).mockResolvedValue(lockedWithPin);
    show();

    expect(await screen.findByRole("heading", { name: "LM Atelier is locked" })).toBeVisible();
    expect(screen.getByLabelText("PIN")).toBeInTheDocument();
    expect(screen.queryByText(/Workspace content/)).toBeNull();
    expect(mounts).toBe(0);
  });

  it("clears what the workspace fetched when it locks, and starts it afresh on unlock", async () => {
    vi.mocked(api.workspaceLockStatus).mockResolvedValueOnce(unlocked).mockResolvedValue(lockedWithoutPin);
    vi.mocked(api.unlockWorkspace).mockResolvedValue(unlocked);
    const client = show();
    expect(await screen.findByText("Workspace content Pottery notes")).toBeVisible();
    expect(client.getQueryData(["notes"])).toBe("Pottery notes");

    act(() => noteWorkspaceLocked());

    expect(screen.queryByText(/Workspace content/)).toBeNull();
    expect(client.getQueryCache().getAll()).toHaveLength(0);
    fireEvent.click(await screen.findByRole("button", { name: "Unlock" }));

    expect(await screen.findByText("Workspace content Pottery notes")).toBeVisible();
    expect(api.unlockWorkspace).toHaveBeenCalledExactlyOnceWith(undefined);
    expect(mounts).toBe(2);
  });

  it("notices an unlock made in another window while it waits", async () => {
    vi.useFakeTimers();
    vi.mocked(api.workspaceLockStatus)
      .mockResolvedValueOnce(lockedWithoutPin)
      .mockResolvedValueOnce(lockedWithoutPin)
      .mockResolvedValue(unlocked);
    show();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByRole("heading", { name: "LM Atelier is locked" })).toBeVisible();
    const reads = vi.mocked(api.workspaceLockStatus).mock.calls.length;

    await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });

    expect(vi.mocked(api.workspaceLockStatus).mock.calls.length).toBe(reads + 1);
    expect(screen.queryByRole("heading", { name: "LM Atelier is locked" })).toBeNull();
    expect(screen.getByText(/Workspace content/)).toBeVisible();
    expect(api.unlockWorkspace).not.toHaveBeenCalled();
  });

  it("still shows the workspace when the lock cannot be read, leaving the refusal to the server", async () => {
    vi.mocked(api.workspaceLockStatus).mockRejectedValue(new Error("neutral read failure"));
    show();

    expect(await screen.findByText(/Workspace content/)).toBeVisible();
  });
});
