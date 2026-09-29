import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StudioRecipes } from "./StudioRecipes";
import { api } from "./api";
import type { EditTemplate } from "./types";

vi.mock("./api", () => ({ api: { editTemplates: vi.fn(), createEditTemplate: vi.fn() } }));

function recipe(overrides: Partial<EditTemplate> = {}): EditTemplate {
  return {
    id: "tpl-1",
    name: "Watercolor",
    description: "",
    instruction: "make it a watercolor painting",
    operation: "image_to_image",
    settings_json: { denoise: 0.42 },
    workflow_revision_id: "rev-1",
    model_profile_id: "profile-1",
    mask_mode: "none",
    trigger_words_json: [],
    content_rating: "general",
    builtin: false,
    enabled: true,
    ...overrides,
  };
}

function renderRecipes(onApply = vi.fn(), from: { runId: string; instruction: string } | null = null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <StudioRecipes onApply={onApply} from={from} />
    </QueryClientProvider>,
  );
  return onApply;
}

describe("StudioRecipes", () => {
  afterEach(cleanup);

  it("applies the whole recipe, not only its words", async () => {
    vi.mocked(api.editTemplates).mockResolvedValue([recipe()]);
    const onApply = renderRecipes();

    fireEvent.click(await screen.findByRole("button", { name: "Watercolor" }));

    // The binding is what makes it a recipe rather than a saved sentence.
    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({ workflow_revision_id: "rev-1", settings_json: { denoise: 0.42 } }),
    );
  });

  it("says which recipes expect a selection before the click", async () => {
    vi.mocked(api.editTemplates).mockResolvedValue([recipe({ mask_mode: "selection" })]);
    renderRecipes();

    expect(await screen.findByText("needs a selection")).toBeInTheDocument();
  });

  it("shows nothing at all when nothing is saved", async () => {
    vi.mocked(api.editTemplates).mockResolvedValue([]);
    const { container } = render(
      <QueryClientProvider client={new QueryClient()}>
        <StudioRecipes onApply={vi.fn()} />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(container).toBeEmptyDOMElement());
  });

  it("leaves a disabled recipe out", async () => {
    vi.mocked(api.editTemplates).mockResolvedValue([recipe({ enabled: false })]);
    renderRecipes();

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Watercolor" })).not.toBeInTheDocument(),
    );
  });

  it("saves the chosen result's edit as a recipe, read from the run that made it", async () => {
    vi.mocked(api.editTemplates).mockResolvedValue([]);
    vi.mocked(api.createEditTemplate).mockResolvedValue(recipe({ id: "tpl-2", name: "Soft watercolor" }));
    renderRecipes(vi.fn(), { runId: "run-7", instruction: "make it a watercolor painting" });

    fireEvent.change(await screen.findByRole("textbox", { name: "Save this edit as a recipe" }), {
      target: { value: "  Soft watercolor " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));

    await waitFor(() =>
      expect(api.createEditTemplate).toHaveBeenCalledWith({
        name: "Soft watercolor",
        instruction: "make it a watercolor painting",
        from_run_id: "run-7",
      }),
    );
    expect(await screen.findByRole("status")).toHaveTextContent("Saved.");
    // The list is asked again, so the new recipe appears with the others.
    await waitFor(() => expect(api.editTemplates).toHaveBeenCalledTimes(2));
  });

  it("saves nothing without a name, and says why a save was refused", async () => {
    vi.mocked(api.editTemplates).mockResolvedValue([]);
    vi.mocked(api.createEditTemplate).mockRejectedValue(new Error("A template with this name already exists."));
    renderRecipes(vi.fn(), { runId: "run-7", instruction: "make it a watercolor painting" });

    fireEvent.click(await screen.findByRole("button", { name: "Save recipe" }));
    expect(api.createEditTemplate).not.toHaveBeenCalled();

    fireEvent.change(screen.getByRole("textbox", { name: "Save this edit as a recipe" }), {
      target: { value: "Watercolor" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("A template with this name already exists.");
  });
});
