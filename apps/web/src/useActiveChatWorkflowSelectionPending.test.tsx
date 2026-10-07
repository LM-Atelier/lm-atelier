import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import type { RoutingMode, WorkflowSelection } from "./types";
import { useActiveChatWorkflowSelection } from "./useActiveChatWorkflowSelection";
import type { ActiveChatWorkflowSelectionState } from "./useActiveChatWorkflowSelection";
import { useShapeAlternatives } from "./shapeAlternatives";
import { WorkflowSelector } from "./WorkflowSelector";

vi.mock("./api", () => ({ api: {
  workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(), setChatWorkflowSelection: vi.fn(),
} }));

type Pending = { finish: () => void; reject: (error: Error) => void };
let pending: Pending[] = [];
let clients: QueryClient[] = [];

beforeEach(() => {
  pending = [];
  clients = [];
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.setChatWorkflowSelection).mockImplementation((_chatId, capability) =>
    new Promise((resolve, reject) => {
      const value: WorkflowSelection = {
        selector_capability: capability, mode: "automatic", workflow_family_id: null,
        workflow_revision_id: null, legacy_profile_id: null,
      };
      pending.push({ finish: () => resolve(value), reject });
    }));
});

afterEach(async () => {
  await act(async () => pending.forEach(request => request.finish()));
  cleanup();
  clients.forEach(client => client.clear());
  vi.clearAllMocks();
});

function client() {
  const value = new QueryClient({ defaultOptions: {
    queries: { retry: false }, mutations: { retry: false },
  } });
  clients.push(value);
  return value;
}

function mount(queryClient: QueryClient, chatId = "chat-one", mode: RoutingMode = "image") {
  const wrapper = ({ children }: { children: ReactNode }) =>
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  return renderHook(({ chatId, mode }: { chatId: string; mode: RoutingMode }) =>
    useActiveChatWorkflowSelection(chatId, mode), { wrapper, initialProps: { chatId, mode } });
}

function ready(state: ActiveChatWorkflowSelectionState) {
  expect(state.kind).toBe("ready");
  if (state.kind !== "ready") throw new Error("Workflow choice is not ready");
  return state;
}

it("accepts one choice before pending notifications render", async () => {
  const { result } = mount(client());
  await waitFor(() => expect(result.current.kind).toBe("ready"));
  const choose = ready(result.current).choose;
  act(() => {
    choose({ mode: "automatic" });
    choose({ mode: "default" });
  });
  await waitFor(() => expect(api.setChatWorkflowSelection).toHaveBeenCalledOnce());
  expect(api.setChatWorkflowSelection).toHaveBeenCalledWith("chat-one", "image", { mode: "automatic" });
});

it("keeps an earlier selector pending after another capability finishes saving", async () => {
  const { result, rerender } = mount(client());
  await waitFor(() => expect(result.current.kind).toBe("ready"));
  act(() => ready(result.current).choose({ mode: "automatic" }));
  await waitFor(() => expect(pending).toHaveLength(1));
  rerender({ chatId: "chat-one", mode: "video" });
  await waitFor(() => expect(result.current.kind).toBe("ready"));
  act(() => ready(result.current).choose({ mode: "default" }));
  await waitFor(() => expect(pending).toHaveLength(2));
  await act(async () => pending[1].finish());
  await waitFor(() => expect(ready(result.current).saving).toBe(false));
  rerender({ chatId: "chat-one", mode: "image" });
  await waitFor(() => expect(result.current.kind).toBe("ready"));
  expect(ready(result.current).saving).toBe(true);
  act(() => ready(result.current).choose({ mode: "default" }));
  expect(api.setChatWorkflowSelection).toHaveBeenCalledTimes(2);
  await act(async () => pending[0].finish());
  await waitFor(() => expect(ready(result.current).saving).toBe(false));
});

it("keeps a remounted selector pending until its existing save reconciles", async () => {
  const shared = client();
  const first = mount(shared);
  await waitFor(() => expect(first.result.current.kind).toBe("ready"));
  act(() => ready(first.result.current).choose({ mode: "automatic" }));
  await waitFor(() => expect(pending).toHaveLength(1));
  first.unmount();
  const second = mount(shared);
  await waitFor(() => expect(second.result.current.kind).toBe("ready"));
  expect(ready(second.result.current).saving).toBe(true);
  act(() => ready(second.result.current).choose({ mode: "default" }));
  expect(api.setChatWorkflowSelection).toHaveBeenCalledOnce();
  await act(async () => pending[0].finish());
  await waitFor(() => expect(ready(second.result.current).saving).toBe(false));
});

it.each([
  { chatId: "chat-two", mode: "image" as const },
  { chatId: "chat-one", mode: "video" as const },
])("lets $chatId/$mode save independently of an active image choice", async ({ chatId, mode }) => {
  const { result, rerender } = mount(client());
  await waitFor(() => expect(result.current.kind).toBe("ready"));
  act(() => ready(result.current).choose({ mode: "automatic" }));
  await waitFor(() => expect(pending).toHaveLength(1));
  rerender({ chatId, mode });
  await waitFor(() => expect(result.current.kind).toBe("ready"));
  expect(ready(result.current).saving).toBe(false);
  act(() => ready(result.current).choose({ mode: "default" }));
  await waitFor(() => expect(api.setChatWorkflowSelection).toHaveBeenCalledTimes(2));
  expect(api.setChatWorkflowSelection).toHaveBeenLastCalledWith(chatId, mode, { mode: "default" });
});

it("lets the same selector retry after its save fails", async () => {
  const { result } = mount(client());
  await waitFor(() => expect(result.current.kind).toBe("ready"));
  act(() => ready(result.current).choose({ mode: "automatic" }));
  await waitFor(() => expect(pending).toHaveLength(1));
  await act(async () => pending[0].reject(new Error("Save failed")));
  await waitFor(() => expect(ready(result.current).saving).toBe(false));
  act(() => ready(result.current).choose({ mode: "default" }));
  await waitFor(() => expect(api.setChatWorkflowSelection).toHaveBeenCalledTimes(2));
});

it.each(["composer", "shape picker"])("shares pending choices from the %s with the other picker", async (first) => {
  const shared = client();
  const wrapper = ({ children }: { children: ReactNode }) =>
    <QueryClientProvider client={shared}>{children}</QueryClientProvider>;
  const { result } = renderHook(() => ({
    active: useActiveChatWorkflowSelection("chat-one", "image"),
    shape: useShapeAlternatives({ chatId: "chat-one", capability: "image", hasAttachments: false,
      families: [], currentRevisionId: null, enabled: true }),
  }), { wrapper });
  await waitFor(() => expect(result.current.active.kind).toBe("ready"));
  act(() => {
    if (first === "composer") ready(result.current.active).choose({ mode: "automatic" });
    else result.current.shape?.onChoose("family-one");
  });
  await waitFor(() => expect(pending).toHaveLength(1));
  await waitFor(() => {
    expect(ready(result.current.active).saving).toBe(true);
    expect(result.current.shape?.choosing).toBe(true);
  });
  act(() => {
    if (first === "composer") result.current.shape?.onChoose("family-two");
    else ready(result.current.active).choose({ mode: "default" });
  });
  expect(api.setChatWorkflowSelection).toHaveBeenCalledOnce();
  await act(async () => pending[0].finish());
  await waitFor(() => {
    expect(ready(result.current.active).saving).toBe(false);
    expect(result.current.shape?.choosing).toBe(false);
  });
});

it("shares pending choices between chat settings and the composer", async () => {
  const shared = client();
  render(<QueryClientProvider client={shared}>
    <WorkflowSelector scope="chat" scopeId="chat-one" capability="image" label="Image workflow" />
  </QueryClientProvider>);
  const { result } = mount(shared);
  await waitFor(() => expect(result.current.kind).toBe("ready"));
  const settings = screen.getByRole("combobox", { name: "Image workflow" });
  await waitFor(() => expect(settings).toBeEnabled());
  fireEvent.change(settings, { target: { value: "automatic" } });
  await waitFor(() => expect(pending).toHaveLength(1));
  await waitFor(() => expect(ready(result.current).saving).toBe(true));
  act(() => ready(result.current).choose({ mode: "default" }));
  expect(api.setChatWorkflowSelection).toHaveBeenCalledOnce();
  await act(async () => pending[0].finish());
  await waitFor(() => expect(ready(result.current).saving).toBe(false));
  act(() => ready(result.current).choose({ mode: "default" }));
  await waitFor(() => expect(pending).toHaveLength(2));
  await waitFor(() => expect(settings).toBeDisabled());
  fireEvent.change(settings, { target: { value: "automatic" } });
  expect(api.setChatWorkflowSelection).toHaveBeenCalledTimes(2);
});
