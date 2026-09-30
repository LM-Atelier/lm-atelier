import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowFamilyPreferences } from "./WorkflowFamilyPreferences";
import { api } from "./api";
import type { WorkflowFamily, WorkflowFamilyPreference } from "./types";

vi.mock("./api", () => ({ api: { setWorkflowFamilyPreference: vi.fn() } }));

const preference: WorkflowFamilyPreference = { selector_capability: "image", enabled: true, is_default: false, sort_order: 2 };
function family(id = "a", isDefault = false): WorkflowFamily {
  return {
    id, name: `Family ${id}`, description: "", use_case: "Neutral scenes", tags: [],
    enabled: true, archived: false, compatibility: false,
    preferences: [{ ...preference, is_default: isDefault }],
    variants: [{ id: `workflow-${id}`, variant_key: "image", name: `Workflow ${id}`, operation: "text_to_image",
      current_revision_id: `revision-${id}`, current_revision_version: 1, engine: "mock", capabilities: ["image"],
      trusted: true, readiness: "ready", readiness_reason: null }],
    created_at: "2026-09-29T00:00:00Z", updated_at: "2026-09-29T00:00:00Z",
  };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}
const clients: QueryClient[] = [];
function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  const tree = (value: WorkflowFamily) => <QueryClientProvider client={client}><WorkflowFamilyPreferences family={value} /></QueryClientProvider>;
  const view = render(tree(family()));
  return { client, changeFamily: (value: WorkflowFamily) => view.rerender(tree(value)) };
}
beforeEach(() => { vi.resetAllMocks(); vi.mocked(api.setWorkflowFamilyPreference).mockResolvedValue(preference); });
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

it("keeps the offered checkbox focused and refuses repeated changes while saving", async () => {
  const pending = deferred<WorkflowFamilyPreference>();
  vi.mocked(api.setWorkflowFamilyPreference).mockReturnValue(pending.promise);
  mount();
  const toggle = screen.getByRole("checkbox", { name: "Images" });
  toggle.focus();
  fireEvent.click(toggle);
  await waitFor(() => expect(toggle).toHaveAttribute("aria-disabled", "true"));
  expect(toggle).not.toBeDisabled();
  expect(toggle).toHaveFocus();
  await act(async () => {
    fireEvent.click(toggle);
    fireEvent.click(screen.getByRole("button", { name: "Make this the default" }));
  });
  expect(api.setWorkflowFamilyPreference).toHaveBeenCalledTimes(1);
  expect(api.setWorkflowFamilyPreference).toHaveBeenCalledWith("a", "image", { enabled: false, is_default: false, sort_order: 2 });
  await act(async () => pending.resolve({ ...preference, enabled: false }));
  expect(toggle).toHaveFocus();
});

it("moves focus from the completed default action to its remaining checkbox", async () => {
  const pending = deferred<WorkflowFamilyPreference>();
  vi.mocked(api.setWorkflowFamilyPreference).mockReturnValue(pending.promise);
  const { changeFamily } = mount();
  const button = screen.getByRole("button", { name: "Make this the default" });
  button.focus();
  fireEvent.click(button);
  await waitFor(() => expect(button).toHaveAttribute("aria-disabled", "true"));
  expect(button).not.toBeDisabled();
  expect(button).toHaveFocus();
  await act(async () => { fireEvent.click(button); });
  expect(api.setWorkflowFamilyPreference).toHaveBeenCalledTimes(1);
  await act(async () => pending.resolve({ ...preference, is_default: true }));
  changeFamily(family("a", true));
  expect(screen.queryByRole("button", { name: "Make this the default" })).not.toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: /Images/ })).toHaveFocus();
});

it("does not steal focus when the user has moved away from the default action", async () => {
  const pending = deferred<WorkflowFamilyPreference>();
  vi.mocked(api.setWorkflowFamilyPreference).mockReturnValue(pending.promise);
  const { changeFamily } = mount();
  const button = screen.getByRole("button", { name: "Make this the default" });
  button.focus();
  fireEvent.click(button);
  await waitFor(() => expect(api.setWorkflowFamilyPreference).toHaveBeenCalledTimes(1));
  const edit = screen.getByRole("button", { name: "Edit family details" });
  edit.focus();
  await act(async () => pending.resolve({ ...preference, is_default: true }));
  changeFamily(family("a", true));
  expect(edit).toHaveFocus();
});

it("keeps controls guarded until the refreshed preferences arrive", async () => {
  const { client } = mount();
  const refreshed = deferred<void>();
  vi.spyOn(client, "invalidateQueries").mockReturnValue(refreshed.promise);
  const toggle = screen.getByRole("checkbox", { name: "Images" });
  fireEvent.click(toggle);
  await waitFor(() => expect(client.invalidateQueries).toHaveBeenCalledTimes(2));
  expect(toggle).toHaveAttribute("aria-disabled", "true");
  await act(async () => { fireEvent.click(toggle); });
  expect(api.setWorkflowFamilyPreference).toHaveBeenCalledTimes(1);
  await act(async () => refreshed.resolve());
  await waitFor(() => expect(toggle).toHaveAttribute("aria-disabled", "false"));
});

it("keeps a failed default action focused and allows retry", async () => {
  vi.mocked(api.setWorkflowFamilyPreference).mockRejectedValueOnce(new Error("Could not save this preference"))
    .mockResolvedValueOnce({ ...preference, is_default: true });
  mount();
  const button = screen.getByRole("button", { name: "Make this the default" });
  button.focus();
  fireEvent.click(button);
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not save this preference");
  expect(button).toHaveFocus();
  expect(button).toHaveAttribute("aria-disabled", "false");
  fireEvent.click(button);
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Images" })).toHaveFocus());
  expect(api.setWorkflowFamilyPreference).toHaveBeenCalledTimes(2);
});

it.each(["resolve", "reject"] as const)("isolates another family while the old save will %s", async (outcome) => {
  const pending = deferred<WorkflowFamilyPreference>();
  vi.mocked(api.setWorkflowFamilyPreference).mockReturnValueOnce(pending.promise).mockResolvedValueOnce(preference);
  const { changeFamily } = mount();
  fireEvent.click(screen.getByRole("checkbox", { name: "Images" }));
  await waitFor(() => expect(api.setWorkflowFamilyPreference).toHaveBeenCalledTimes(1));
  changeFamily(family("b"));
  const other = screen.getByRole("checkbox", { name: "Images" });
  expect(other).not.toBeDisabled();
  expect(other).toHaveAttribute("aria-disabled", "false");
  other.focus();
  fireEvent.click(other);
  await waitFor(() => expect(api.setWorkflowFamilyPreference).toHaveBeenNthCalledWith(2, "b", "image", {
    enabled: false, is_default: false, sort_order: 2,
  }));
  await act(async () => {
    if (outcome === "resolve") pending.resolve(preference);
    else pending.reject(new Error("Earlier family save failed"));
  });
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(other).toHaveFocus();
  expect(screen.getByRole("heading", { name: "Family b" })).toBeInTheDocument();
});
