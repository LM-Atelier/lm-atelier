import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { GenerationEditRecipeDialog } from "./GenerationEditRecipeDialog";
import type { EditTemplate } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, editRecipeDraft: vi.fn(), createEditTemplate: vi.fn() } };
});

const SAVED = { id: "tmpl_1", name: "Bluer cup" } as EditTemplate;

function show(onClose = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><GenerationEditRecipeDialog runId="run_1" onClose={onClose} /></QueryClientProvider>);
  return onClose;
}

async function nameIt(name: string) {
  const dialog = await screen.findByRole("dialog", { name: "Keep this edit as a recipe" });
  fireEvent.change(await within(dialog).findByRole("textbox", { name: "Recipe name" }), { target: { value: name } });
  return dialog;
}

beforeEach(() => {
  vi.mocked(api.editRecipeDraft).mockResolvedValue({ run_id: "run_1", instruction: "make the cup blue" });
  vi.mocked(api.createEditTemplate).mockResolvedValue(SAVED);
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("starts from the words the edit ran with and saves them under the name given, from that run", async () => {
  show();
  const dialog = await nameIt(" Bluer cup ");

  expect(within(dialog).getByRole("textbox", { name: "Words" })).toHaveValue("make the cup blue");
  fireEvent.click(within(dialog).getByRole("button", { name: "Save recipe" }));

  expect(await within(dialog).findByText(/Saved the recipe Bluer cup\./)).toBeInTheDocument();
  expect(api.createEditTemplate).toHaveBeenCalledExactlyOnceWith({
    name: "Bluer cup", instruction: "make the cup blue", from_run_id: "run_1",
  });
  // The form is gone with the saved recipe, so focus moves to what comes next.
  expect(within(dialog).getByRole("button", { name: "Done" })).toHaveFocus();
});

it("saves the words as changed, and nothing without a name", async () => {
  show();
  // Blank rather than empty, so the dialog's own check is what refuses it.
  const dialog = await nameIt("   ");
  fireEvent.change(within(dialog).getByRole("textbox", { name: "Words" }), { target: { value: "make the cup red" } });

  fireEvent.click(within(dialog).getByRole("button", { name: "Save recipe" }));
  expect(api.createEditTemplate).not.toHaveBeenCalled();
  fireEvent.change(within(dialog).getByRole("textbox", { name: "Recipe name" }), { target: { value: "Red cup" } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Save recipe" }));

  await waitFor(() => expect(api.createEditTemplate).toHaveBeenCalledExactlyOnceWith({
    name: "Red cup", instruction: "make the cup red", from_run_id: "run_1",
  }));
});

it("says why a recipe was not saved and keeps what was typed", async () => {
  vi.mocked(api.createEditTemplate).mockRejectedValue(
    new ApiError(409, "A template with this name already exists.", "A template with this name already exists.",
      "edit-template-name-taken"),
  );
  show();
  const dialog = await nameIt("Bluer cup");

  fireEvent.click(within(dialog).getByRole("button", { name: "Save recipe" }));

  expect(await within(dialog).findByRole("alert")).toHaveTextContent("A template with this name already exists.");
  expect(within(dialog).getByRole("textbox", { name: "Recipe name" })).toHaveValue("Bluer cup");
});

it("says so when the edit cannot be read, and offers nothing to save", async () => {
  vi.mocked(api.editRecipeDraft).mockRejectedValue(
    new ApiError(404, "This generation no longer exists.", "This generation no longer exists.", "output-recipe-run-not-found"));
  show();

  const dialog = await screen.findByRole("dialog", { name: "Keep this edit as a recipe" });

  expect(await within(dialog).findByText(/This edit could not be read/)).toBeInTheDocument();
  expect(within(dialog).queryByRole("button", { name: "Save recipe" })).toBeNull();
});
