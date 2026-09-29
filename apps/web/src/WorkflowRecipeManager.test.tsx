import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { WorkflowRecipeManager } from "./WorkflowRecipeManager";
import type { WorkflowUseCasePreset } from "./workflowUseCaseTypes";

vi.mock("./api", () => ({ api: {
  workflowUseCasePresets: vi.fn(), projects: vi.fn(), workflowSummaries: vi.fn(), engines: vi.fn(),
  createWorkflowUseCasePreset: vi.fn(), replaceWorkflowUseCasePreset: vi.fn(), deleteWorkflowUseCasePreset: vi.fn(),
  workflowUseCaseDefault: vi.fn(), setWorkflowUseCaseDefault: vi.fn(), workflowUseCaseChoice: vi.fn(), setWorkflowUseCaseChoice: vi.fn(),
} }));
const clients: QueryClient[] = [];
const recipe: WorkflowUseCasePreset = { id: "saved", name: "Detailed", use_case: "image_generation", settings_json: {}, enabled: true, is_default: false, builtin: false };
let recipes: WorkflowUseCasePreset[];
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><WorkflowRecipeManager onClose={vi.fn()} /></QueryClientProvider>);
}
beforeEach(() => {
  recipes = [recipe];
  vi.mocked(api.workflowUseCasePresets).mockImplementation(async () => recipes);
  vi.mocked(api.projects).mockResolvedValue([]);
  vi.mocked(api.workflowSummaries).mockResolvedValue([]);
  vi.mocked(api.engines).mockResolvedValue([]);
  vi.mocked(api.workflowUseCaseDefault).mockResolvedValue({ preset_id: null });
  vi.mocked(api.workflowUseCaseChoice).mockResolvedValue({ mode: "inherit" });
  vi.mocked(api.createWorkflowUseCasePreset).mockImplementation(async (payload) => {
    const next = { ...payload, id: "new", builtin: false }; recipes = [...recipes, next]; return next;
  });
  vi.mocked(api.replaceWorkflowUseCasePreset).mockImplementation(async (id, payload) => {
    const next = { ...payload, id, builtin: false }; recipes = recipes.map((item) => item.id === id ? next : item); return next;
  });
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); vi.resetAllMocks(); });

it("creates and edits recipes and refreshes the visible list after each save", async () => {
  show();
  await screen.findByRole("button", { name: "Edit recipe Detailed" });
  fireEvent.click(screen.getByRole("button", { name: "New recipe" }));
  expect(screen.getByRole("textbox", { name: "Recipe name" })).toHaveFocus();
  fireEvent.change(screen.getByRole("textbox", { name: "Recipe name" }), { target: { value: "Another recipe" } });
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  await screen.findByRole("button", { name: "Edit recipe Another recipe" });
  expect(screen.getByRole("button", { name: "New recipe" })).toHaveFocus();
  fireEvent.click(screen.getByRole("button", { name: "Edit recipe Another recipe" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Recipe name" }), { target: { value: "Renamed" } });
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  await screen.findByRole("button", { name: "Edit recipe Renamed" });
  expect(api.replaceWorkflowUseCasePreset).toHaveBeenCalledWith("new", expect.objectContaining({ name: "Renamed" }));
});

it("restores focus inside the dialog after a successful deletion", async () => {
  vi.mocked(api.deleteWorkflowUseCasePreset).mockImplementation(async () => { recipes = []; });
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Delete recipe Detailed" }));
  const confirm = screen.getByRole("button", { name: "Confirm deletion" });
  expect(confirm).toHaveFocus();
  fireEvent.click(confirm);
  await screen.findByText("Recipe deleted.");
  expect(screen.getByRole("button", { name: "New recipe" })).toHaveFocus();
});

it("keeps a failed save open and retains the entered name", async () => {
  vi.mocked(api.replaceWorkflowUseCasePreset).mockRejectedValue(new Error("Recipe changed. Refresh and try again."));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Edit recipe Detailed" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Recipe name" }), { target: { value: "Retained draft" } });
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  await screen.findByText("Recipe changed. Refresh and try again.");
  expect(screen.getByRole("textbox", { name: "Recipe name" })).toHaveValue("Retained draft");
});

it("preserves an in-use recipe when deletion is refused", async () => {
  vi.mocked(api.deleteWorkflowUseCasePreset).mockRejectedValue(new Error("Remove selections before deleting this recipe."));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Delete recipe Detailed" }));
  expect(api.deleteWorkflowUseCasePreset).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Confirm deletion" }));
  await screen.findByText("Remove selections before deleting this recipe.");
  expect(screen.getByRole("button", { name: "Edit recipe Detailed" })).toBeVisible();
});

it("hides edits and deletion for built-in recipes", async () => {
  recipes = [{ ...recipe, builtin: true }];
  show();
  await screen.findByText("Detailed");
  expect(screen.queryByRole("button", { name: "Edit recipe Detailed" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Delete recipe Detailed" })).toBeNull();
});

it("does not present failed catalog or project reads as empty lists", async () => {
  vi.mocked(api.workflowUseCasePresets).mockRejectedValue(new Error("Catalog unavailable"));
  vi.mocked(api.projects).mockRejectedValue(new Error("Projects unavailable"));
  show();
  await screen.findByText("Catalog unavailable");
  expect(screen.queryByText(/No recipes yet/)).toBeNull();
  fireEvent.click(screen.getByText("Project choices"));
  await screen.findByText("Projects unavailable");
  expect(screen.getByRole("combobox", { name: "Project for recipe choices" })).toBeDisabled();
  expect(screen.queryByRole("option", { name: "Choose a project" })).toBeNull();
});

it("offers workspace defaults separately from project inheritance", async () => {
  vi.mocked(api.projects).mockResolvedValue([{ id: "p", name: "Project", description: "", instructions: "", archived: false, pinned: false,
    image_workflow_revision_id: null, video_workflow_revision_id: null, created_at: "", updated_at: "" }]);
  show();
  fireEvent.click(screen.getByText("Workspace defaults"));
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Image generation recipe" })).toHaveValue("automatic"));
  fireEvent.click(screen.getByText("Workspace defaults"));
  fireEvent.click(screen.getByText("Project choices"));
  await screen.findByRole("option", { name: "Project" });
  fireEvent.change(screen.getByRole("combobox", { name: "Project for recipe choices" }), { target: { value: "p" } });
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Image generation recipe" })).toHaveValue("inherit"));
});
