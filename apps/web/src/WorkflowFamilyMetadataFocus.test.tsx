import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowFamilyPreferences } from "./WorkflowFamilyPreferences";
import { api } from "./api";
import type { WorkflowFamily } from "./types";

vi.mock("./api", () => ({ api: { updateWorkflowFamily: vi.fn(), setWorkflowFamilyPreference: vi.fn() } }));

function family(id = "a"): WorkflowFamily {
  return {
    id, name: `Family ${id}`, description: "", use_case: "Neutral scenes", tags: [],
    enabled: true, archived: false, compatibility: false, preferences: [], variants: [],
    created_at: "2026-09-29T00:00:00Z", updated_at: "2026-09-29T00:00:00Z",
  };
}
const clients: QueryClient[] = [];
function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  const tree = (value: WorkflowFamily) => <QueryClientProvider client={client}><WorkflowFamilyPreferences family={value} /></QueryClientProvider>;
  const view = render(tree(family()));
  return (value: WorkflowFamily) => view.rerender(tree(value));
}
function open() {
  fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
  return screen.getByRole("textbox", { name: "Family name" });
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.updateWorkflowFamily).mockResolvedValue(family());
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

it("focuses the name when opening and returns to Edit after cancellation", () => {
  mount();
  expect(screen.getByRole("button", { name: "Edit family details" })).not.toHaveFocus();
  const name = open();
  expect(name).toHaveFocus();
  fireEvent.change(name, { target: { value: "Discarded draft" } });
  const cancel = screen.getByRole("button", { name: "Cancel family changes" });
  cancel.focus();
  fireEvent.click(cancel);
  expect(screen.getByRole("button", { name: "Edit family details" })).toHaveFocus();
  expect(api.updateWorkflowFamily).not.toHaveBeenCalled();
  expect(open()).toHaveValue("Family a");
});

it.each(["button", "name"])("keeps %s focus and prevents repeat actions during a save", async (target) => {
  let finish: ((value: WorkflowFamily) => void) | undefined;
  vi.mocked(api.updateWorkflowFamily).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  mount();
  const name = open();
  const save = screen.getByRole("button", { name: "Save family details" });
  const cancel = screen.getByRole("button", { name: "Cancel family changes" });
  const focused = target === "button" ? save : name;
  focused.focus();
  if (target === "button") fireEvent.click(save);
  else fireEvent.submit(name.closest("form")!);
  await waitFor(() => expect(save).toHaveAttribute("aria-disabled", "true"));
  expect(save).not.toBeDisabled();
  expect(cancel).not.toBeDisabled();
  expect(cancel).toHaveAttribute("aria-disabled", "true");
  expect(name).toHaveAttribute("readonly");
  expect(screen.getByRole("textbox", { name: "Use case" })).toHaveAttribute("readonly");
  expect(focused).toHaveFocus();
  await act(async () => {
    fireEvent.click(save);
    fireEvent.submit(name.closest("form")!);
    fireEvent.click(cancel);
  });
  expect(api.updateWorkflowFamily).toHaveBeenCalledTimes(1);
  expect(name).toBeInTheDocument();
  await act(async () => finish?.(family()));
  await waitFor(() => expect(screen.getByRole("button", { name: "Edit family details" })).toHaveFocus());
});

it("keeps Save focused after failure and returns to Edit after retry", async () => {
  vi.mocked(api.updateWorkflowFamily).mockRejectedValueOnce(new Error("Could not save family details"))
    .mockResolvedValueOnce({ ...family(), name: "Retained name" });
  mount();
  fireEvent.change(open(), { target: { value: "Retained name" } });
  const save = screen.getByRole("button", { name: "Save family details" });
  save.focus();
  fireEvent.click(save);
  await screen.findByRole("alert");
  expect(save).toHaveFocus();
  expect(save).toHaveAttribute("aria-disabled", "false");
  expect(screen.getByRole("textbox", { name: "Family name" })).not.toHaveAttribute("readonly");
  fireEvent.click(save);
  await waitFor(() => expect(screen.getByRole("button", { name: "Edit family details" })).toHaveFocus());
  expect(api.updateWorkflowFamily).toHaveBeenNthCalledWith(2, "a", { name: "Retained name" });
});

it("keeps an invalid Save reachable while refusing both click and form submission", async () => {
  mount();
  const name = open();
  fireEvent.change(name, { target: { value: "   " } });
  const save = screen.getByRole("button", { name: "Save family details" });
  save.focus();
  expect(save).toHaveFocus();
  expect(save).not.toBeDisabled();
  expect(save).toHaveAttribute("aria-disabled", "true");
  await act(async () => { fireEvent.click(save); fireEvent.submit(name.closest("form")!); });
  expect(api.updateWorkflowFamily).not.toHaveBeenCalled();
});

it("does not move focus out of another family's draft when an earlier save finishes", async () => {
  let finish: ((value: WorkflowFamily) => void) | undefined;
  vi.mocked(api.updateWorkflowFamily).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const changeFamily = mount();
  open();
  fireEvent.click(screen.getByRole("button", { name: "Save family details" }));
  await waitFor(() => expect(api.updateWorkflowFamily).toHaveBeenCalledTimes(1));
  changeFamily(family("b"));
  const name = open();
  fireEvent.change(name, { target: { value: "Other draft" } });
  expect(name).toHaveFocus();
  await act(async () => finish?.(family()));
  expect(name).toHaveFocus();
  expect(name).toHaveValue("Other draft");
  expect(api.updateWorkflowFamily).toHaveBeenCalledTimes(1);
});
