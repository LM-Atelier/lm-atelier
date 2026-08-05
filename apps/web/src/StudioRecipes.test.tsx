import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StudioRecipes } from "./StudioRecipes";
import { api } from "./api";
import type { EditTemplate } from "./types";

vi.mock("./api", () => ({ api: { editTemplates: vi.fn() } }));

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

function renderRecipes(onApply = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <StudioRecipes onApply={onApply} />
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
});
