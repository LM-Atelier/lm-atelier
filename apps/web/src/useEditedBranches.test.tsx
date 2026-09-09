import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { useEditedBranches } from "./useEditedBranches";
import type { ChatDetail, EditedBranch } from "./types";

const chat: ChatDetail = {
  id: "chat-branch", project_id: null, title: "Blue boat", pinned: false, archived: false,
  routing_mode: "text", confirm_uncertain_media: false, active_chat_profile_id: null,
  active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: "original",
  created_at: "2026-01-01", updated_at: "2026-01-01", messages: [],
};
const branch: EditedBranch = {
  source_message_id: "source", source_run_id: "source-run", branch_head_message_id: "edited",
  source_available: true, can_continue: true, jobs: [],
  plan: {
    id: "plan", chat_id: chat.id, idempotency_key: "accepted-edit", source_action: "edit_and_branch",
    persistence_scope: "durable", status: "complete", context_head_message_id: null,
    transcript_sequence: 8, priority: 0, planner_version: "single-turn-v1", failure_policy: "stop",
    summary_json: {}, steps: [], created_at: "2026-01-01", updated_at: "2026-01-01",
  },
};

function Probe({ current = chat }: { current?: ChatDetail }) {
  const edited = useEditedBranches(current);
  return <div>
    <span data-testid="count">{edited.branches.length}</span>
    <span data-testid="preview">{edited.preview?.branch_head_message_id ?? "closed"}</span>
    <span data-testid="error">{String(edited.failed || edited.activationFailed)}</span>
    <button onClick={() => edited.view(branch)}>View</button>
    <button onClick={() => edited.continueBranch(branch)}>Continue</button>
  </div>;
}

function mount() {
  const client = new QueryClient({ defaultOptions: {
    queries: { retry: false }, mutations: { retry: false },
  } });
  client.setQueryData(["chat", chat.id], chat);
  const rendered = render(<QueryClientProvider client={client}><Probe /></QueryClientProvider>);
  return { client, ...rendered };
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("loads every branch page with cancellation and keeps preview local", async () => {
  const pages = vi.spyOn(api, "editedBranches").mockImplementation(async (_chat, cursor) => cursor
    ? { items: [{ ...branch, plan: { ...branch.plan, id: "older" } }], next_cursor: null }
    : { items: [branch], next_cursor: "older-cursor" });
  const activate = vi.spyOn(api, "activateEditedBranch");
  const { client } = mount();
  await waitFor(() => expect(screen.getByTestId("count")).toHaveTextContent("2"));
  expect(pages).toHaveBeenNthCalledWith(1, chat.id, null, expect.any(AbortSignal));
  expect(pages).toHaveBeenNthCalledWith(2, chat.id, "older-cursor", expect.any(AbortSignal));
  fireEvent.click(screen.getByText("View"));
  expect(screen.getByTestId("preview")).toHaveTextContent("edited");
  expect(client.getQueryData<ChatDetail>(["chat", chat.id])?.active_head_message_id).toBe("original");
  expect(activate).not.toHaveBeenCalled();
});

it("activates with the observed head and updates only the returned chat", async () => {
  vi.spyOn(api, "editedBranches").mockResolvedValue({ items: [branch], next_cursor: null });
  const activate = vi.spyOn(api, "activateEditedBranch").mockResolvedValue({
    chat_id: chat.id, active_head_message_id: "edited",
  });
  const { client } = mount();
  client.setQueryData(["chat", "other"], { ...chat, id: "other", active_head_message_id: "other-head" });
  fireEvent.click(screen.getByText("View"));
  fireEvent.click(screen.getByText("Continue"));
  await waitFor(() => expect(screen.getByTestId("preview")).toHaveTextContent("closed"));
  expect(activate).toHaveBeenCalledWith(chat.id, "plan", "original");
  expect(client.getQueryData<ChatDetail>(["chat", chat.id])?.active_head_message_id).toBe("edited");
  expect(client.getQueryData<ChatDetail>(["chat", "other"])?.active_head_message_id).toBe("other-head");
});

it("keeps the active conversation and preview on a rejected activation", async () => {
  vi.spyOn(api, "editedBranches").mockResolvedValue({ items: [branch], next_cursor: null });
  vi.spyOn(api, "activateEditedBranch").mockRejectedValue(new Error("Head changed"));
  const { client } = mount();
  fireEvent.click(screen.getByText("View"));
  fireEvent.click(screen.getByText("Continue"));
  await waitFor(() => expect(screen.getByTestId("error")).toHaveTextContent("true"));
  expect(screen.getByTestId("preview")).toHaveTextContent("edited");
  expect(client.getQueryData<ChatDetail>(["chat", chat.id])?.active_head_message_id).toBe("original");
});

it("rejects a repeated page cursor instead of silently truncating or looping", async () => {
  const pages = vi.spyOn(api, "editedBranches").mockResolvedValue({
    items: [branch], next_cursor: "same",
  });
  mount();
  await waitFor(() => expect(screen.getByTestId("error")).toHaveTextContent("true"));
  expect(pages).toHaveBeenCalledTimes(2);
  expect(screen.getByTestId("count")).toHaveTextContent("0");
});

it("does not carry a preview into another chat", async () => {
  vi.spyOn(api, "editedBranches").mockResolvedValue({ items: [branch], next_cursor: null });
  const { client, rerender } = mount();
  fireEvent.click(screen.getByText("View"));
  await act(async () => rerender(<QueryClientProvider client={client}>
    <Probe current={{ ...chat, id: "other" }} />
  </QueryClientProvider>));
  expect(screen.getByTestId("preview")).toHaveTextContent("closed");
});
