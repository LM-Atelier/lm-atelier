import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { api } from "./api";
import { ModelUpdatesPanel } from "./ModelUpdatesPanel";
import type { CatalogModel, ModelUpdate } from "./types";

vi.mock("./api", () => ({
  api: {
    modelUpdates: vi.fn(),
    catalogItemDetail: vi.fn(),
  },
}));

const update = (overrides: Partial<ModelUpdate> = {}): ModelUpdate => ({
  install_id: "asset-1",
  name: "portrait-lora",
  kind: "lora",
  model_id: "101",
  installed_version_id: "201",
  installed_version_name: "v1",
  state: "update_available",
  update_version_id: "204",
  update_version_name: "v4",
  update_published_at: "2026-07-30T00:00:00Z",
  update_base_model: "SDXL 1.0",
  update_changelog: "Sharper hands",
  ...overrides,
});

function renderPanel(onInstall = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <ModelUpdatesPanel onInstall={onInstall} />
    </QueryClientProvider>,
  );
  return onInstall;
}

describe("ModelUpdatesPanel", () => {
  beforeEach(() => {
    vi.mocked(api.modelUpdates).mockResolvedValue([update()]);
  });
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("asks the provider only when the button is pressed", async () => {
    renderPanel();
    expect(api.modelUpdates).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /Check for updates/ }));

    await screen.findByText("portrait-lora");
    expect(api.modelUpdates).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/1 update available/)).toBeInTheDocument();
    expect(screen.getByText("Sharper hands")).toBeInTheDocument();
  });

  it("separates unreachable checks from verdicts", async () => {
    vi.mocked(api.modelUpdates).mockResolvedValue([
      update({ state: "current", update_version_id: null }),
      update({
        install_id: "asset-2",
        name: "mystery-lora",
        state: "unknown",
        update_version_id: null,
      }),
    ]);
    renderPanel();

    fireEvent.click(screen.getByRole("button", { name: /Check for updates/ }));

    expect(await screen.findByText(/Everything checkable is up to date/)).toBeInTheDocument();
    expect(screen.getByText(/1 unreachable/)).toBeInTheDocument();
    expect(screen.getByText(/Could not check: mystery-lora/)).toBeInTheDocument();
  });

  it("routes an update into the normal verified install flow", async () => {
    const model = { provider: "civitai", remote_id: "204" } as CatalogModel;
    vi.mocked(api.catalogItemDetail).mockResolvedValue({
      model,
      revision: "204",
      files: [],
    });
    const onInstall = renderPanel();

    fireEvent.click(screen.getByRole("button", { name: /Check for updates/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Review update" }));

    await waitFor(() =>
      expect(api.catalogItemDetail).toHaveBeenCalledWith("civitai", "204", "lora"));
    await waitFor(() => expect(onInstall).toHaveBeenCalledWith(model, "lora"));
  });

  it("reports when nothing names an exact version", async () => {
    vi.mocked(api.modelUpdates).mockResolvedValue([]);
    renderPanel();

    fireEvent.click(screen.getByRole("button", { name: /Check for updates/ }));

    expect(
      await screen.findByText(/Nothing installed names an exact provider version/),
    ).toBeInTheDocument();
  });
});
