import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { WorkflowRecipeChoices, ChatWorkflowRecipes } from "./WorkflowRecipeChoices";
import type { WorkflowRecipeScope, WorkflowUseCaseChoice, WorkflowUseCasePreset } from "./workflowUseCaseTypes";

vi.mock("./api", () => ({ api: {
  workflowUseCasePresets: vi.fn(), workflowUseCaseChoice: vi.fn(),
  setWorkflowUseCaseChoice: vi.fn(), workflowUseCaseDefault: vi.fn(),
  setWorkflowUseCaseDefault: vi.fn(),
} }));
const clients: QueryClient[] = [];
const recipe: WorkflowUseCasePreset = {
  id: "fine", name: "Fine detail", use_case: "image_generation",
  settings_json: { steps: 24 }, enabled: true, is_default: false, builtin: false,
};
const scope: WorkflowRecipeScope = { kind: "chat", id: "chat-a" };
let choices: Record<string, WorkflowUseCaseChoice>;
function show(selectedScope = scope) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  const view = render(<QueryClientProvider client={client}>
    <WorkflowRecipeChoices scope={selectedScope} />
  </QueryClientProvider>);
  return { client, ...view, changeScope: (next: WorkflowRecipeScope) => view.rerender(
    <QueryClientProvider client={client}><WorkflowRecipeChoices scope={next} /></QueryClientProvider>,
  ) };
}
beforeEach(() => {
  choices = { image_edit: { mode: "automatic" }, image_generation: { mode: "preset", preset_id: "fine" } };
  vi.mocked(api.workflowUseCasePresets).mockResolvedValue([recipe]);
  vi.mocked(api.workflowUseCaseChoice).mockImplementation(async (_scope, useCase) => choices[useCase] ?? { mode: "inherit" });
  vi.mocked(api.setWorkflowUseCaseChoice).mockImplementation(async (_scope, useCase, choice) => {
    choices[useCase] = choice;
    return choice;
  });
  vi.mocked(api.workflowUseCaseDefault).mockResolvedValue({ preset_id: null });
  vi.mocked(api.setWorkflowUseCaseDefault).mockImplementation(async (_useCase, value) => value);
});
afterEach(() => { cleanup(); for (const client of clients.splice(0)) client.clear(); vi.resetAllMocks(); });

it("keeps all eight choices independent and distinguishes inheritance from Automatic", async () => {
  show();
  const image = screen.getByRole("combobox", { name: "Image generation recipe" });
  await waitFor(() => expect(image).toHaveValue("preset:fine"));
  expect(screen.getAllByRole("combobox")).toHaveLength(8);
  const edit = screen.getByRole("combobox", { name: "Whole-image edit recipe" });
  expect(edit).toHaveValue("automatic");
  expect(screen.getByRole("combobox", { name: "Chat recipe" })).toHaveValue("inherit");
  fireEvent.change(image, { target: { value: "automatic" } });
  await waitFor(() => expect(api.setWorkflowUseCaseChoice).toHaveBeenCalledExactlyOnceWith(scope, "image_generation", { mode: "automatic" }));
  await waitFor(() => expect(image).toHaveValue("automatic"));
  expect(edit).toHaveValue("automatic");
  fireEvent.change(image, { target: { value: "inherit" } });
  await waitFor(() => expect(image).toHaveValue("inherit"));
  expect(api.setWorkflowUseCaseChoice).toHaveBeenLastCalledWith(scope, "image_generation", { mode: "inherit" });
  expect(within(edit).queryByRole("option", { name: "Fine detail" })).toBeNull();
});

it("leaves a failed read unresolved and retries only after an explicit action", async () => {
  vi.mocked(api.workflowUseCaseChoice).mockImplementation(async (_scope, useCase) => {
    if (useCase === "image_edit") throw new Error("Choice unavailable");
    return { mode: "inherit" };
  });
  show();
  await screen.findByText("Choice unavailable");
  const edit = screen.getByRole("combobox", { name: "Whole-image edit recipe" });
  expect(edit).toHaveValue("");
  expect(edit).toBeDisabled();
  expect(screen.getByRole("combobox", { name: "Chat recipe" })).toHaveValue("inherit");
  expect(api.setWorkflowUseCaseChoice).not.toHaveBeenCalled();
  vi.mocked(api.workflowUseCaseChoice).mockResolvedValue({ mode: "automatic" });
  fireEvent.click(screen.getByRole("button", { name: "Retry Whole-image edit recipe" }));
  await waitFor(() => expect(edit).toHaveValue("automatic"));
});

it("retains unavailable selections without offering disabled recipes as new choices", async () => {
  choices.image_generation = { mode: "preset", preset_id: "retired" };
  vi.mocked(api.workflowUseCasePresets).mockResolvedValue([{ ...recipe, id: "retired", enabled: false }]);
  show();
  const image = screen.getByRole("combobox", { name: "Image generation recipe" });
  await waitFor(() => expect(image).toHaveValue("preset:retired"));
  expect(within(image).getByRole("option", { name: "Fine detail (unavailable)" })).toBeDisabled();
  expect(api.setWorkflowUseCaseChoice).not.toHaveBeenCalled();
});

it("does not reuse cached choices after a failed refresh", async () => {
  const { client } = show();
  const image = screen.getByRole("combobox", { name: "Image generation recipe" });
  await waitFor(() => expect(image).toHaveValue("preset:fine"));
  vi.mocked(api.workflowUseCaseChoice).mockRejectedValue(new Error("Refresh failed"));
  await act(async () => { await client.invalidateQueries({ queryKey: ["workflow-recipe-choice"] }); });
  await waitFor(() => expect(image).toHaveValue(""));
  expect(image).toBeDisabled();
});

it("keeps focus while saving and restores the confirmed choice on failure", async () => {
  let rejectSave!: (error: Error) => void;
  vi.mocked(api.setWorkflowUseCaseChoice).mockImplementation(() => new Promise((_resolve, reject) => { rejectSave = reject; }));
  show();
  const image = screen.getByRole("combobox", { name: "Image generation recipe" });
  await waitFor(() => expect(image).toHaveValue("preset:fine"));
  image.focus();
  fireEvent.change(image, { target: { value: "automatic" } });
  await waitFor(() => expect(image).toHaveAttribute("aria-disabled", "true"));
  expect(image).toHaveFocus();
  fireEvent.change(image, { target: { value: "inherit" } });
  expect(api.setWorkflowUseCaseChoice).toHaveBeenCalledTimes(1);
  await act(async () => { rejectSave(new Error("Save failed")); });
  await screen.findByText("Save failed");
  expect(image).toHaveValue("preset:fine");
  expect(image).toHaveFocus();
});

it("keeps an old chat save from changing the newly selected chat", async () => {
  let finishSave!: (choice: WorkflowUseCaseChoice) => void;
  vi.mocked(api.setWorkflowUseCaseChoice).mockImplementation(() => new Promise((resolve) => { finishSave = resolve; }));
  const { changeScope } = show();
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Image generation recipe" })).toHaveValue("preset:fine"));
  fireEvent.change(screen.getByRole("combobox", { name: "Image generation recipe" }), { target: { value: "automatic" } });
  await waitFor(() => expect(api.setWorkflowUseCaseChoice).toHaveBeenCalledTimes(1));
  changeScope({ kind: "chat", id: "chat-b" });
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Image generation recipe" })).toHaveValue("preset:fine"));
  await act(async () => { finishSave({ mode: "automatic" }); });
  expect(screen.getByRole("combobox", { name: "Image generation recipe" })).toHaveValue("preset:fine");
});

it("writes project inheritance separately from workspace defaults", async () => {
  const { changeScope } = show({ kind: "project", id: "project-a" });
  let image = screen.getByRole("combobox", { name: "Image generation recipe" });
  await waitFor(() => expect(image).toHaveValue("preset:fine"));
  fireEvent.change(image, { target: { value: "inherit" } });
  await waitFor(() => expect(api.setWorkflowUseCaseChoice).toHaveBeenCalledWith({ kind: "project", id: "project-a" }, "image_generation", { mode: "inherit" }));
  changeScope({ kind: "workspace" });
  image = screen.getByRole("combobox", { name: "Image generation recipe" });
  await waitFor(() => expect(image).toHaveValue("automatic"));
  expect(within(image).queryByRole("option", { name: /Inherit/ })).toBeNull();
  fireEvent.change(image, { target: { value: "preset:fine" } });
  await waitFor(() => expect(api.setWorkflowUseCaseDefault).toHaveBeenCalledExactlyOnceWith("image_generation", { preset_id: "fine" }));
});

it("loads every catalog page before offering recipes", async () => {
  const firstPage = Array.from({ length: 200 }, (_, i) => ({ ...recipe, id: `first-${i}`, name: `Recipe ${i}` }));
  vi.mocked(api.workflowUseCasePresets).mockImplementation(async (_useCase, offset) => offset === 0 ? firstPage : [recipe]);
  show();
  const image = screen.getByRole("combobox", { name: "Image generation recipe" });
  await waitFor(() => expect(within(image).getByRole("option", { name: "Fine detail" })).toBeEnabled());
  expect(api.workflowUseCasePresets).toHaveBeenCalledWith(undefined, 200, expect.any(AbortSignal));
});

it("never presents a partial catalog when a later page fails", async () => {
  vi.mocked(api.workflowUseCasePresets).mockImplementation(async (_useCase, offset) => {
    if (offset !== 0) throw new Error("Next page failed");
    return Array.from({ length: 200 }, (_, i) => ({ ...recipe, id: `first-${i}` }));
  });
  show();
  await screen.findAllByText("Next page failed");
  expect(screen.getByRole("combobox", { name: "Image generation recipe" })).toBeDisabled();
  expect(screen.queryByRole("option", { name: "Fine detail" })).toBeNull();
});

it("refuses a repeated catalog page instead of fetching indefinitely", async () => {
  vi.mocked(api.workflowUseCasePresets).mockResolvedValue(Array.from({ length: 200 }, (_, i) => ({ ...recipe, id: `same-${i}` })));
  show();
  await screen.findAllByText("The recipe list changed while loading. Try again.");
  expect(screen.getByRole("combobox", { name: "Image generation recipe" })).toBeDisabled();
  expect(api.workflowUseCasePresets).toHaveBeenCalledTimes(2);
});

it("does not fetch recipes until the chat panel opens", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><ChatWorkflowRecipes chatId="chat-a" /></QueryClientProvider>);
  expect(api.workflowUseCasePresets).not.toHaveBeenCalled();
  const summary = screen.getByText("Recipes for this chat");
  fireEvent.click(summary);
  fireEvent(summary.parentElement!, new Event("toggle"));
  await waitFor(() => expect(api.workflowUseCasePresets).toHaveBeenCalled());
  expect(screen.getAllByRole("combobox")).toHaveLength(8);
});
