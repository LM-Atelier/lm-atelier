import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { catalogUnavailableMessage } from "./catalogSourceMessages";
import { ModelsView } from "./ModelsView";

vi.mock("./ModelUpdatesPanel", () => ({ ModelUpdatesPanel: () => null }));
vi.mock("./useCatalogInstall", () => ({
  useCatalogInstall: () => ({
    pendingInstall: null,
    cancel: vi.fn(),
    prepare: { isPending: false, variables: undefined, mutate: vi.fn() },
    confirm: { isPending: false, mutate: vi.fn() },
  }),
}));
vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      catalog: vi.fn(),
      recipes: vi.fn(),
      models: vi.fn(),
      modelAssets: vi.fn(),
      jobs: vi.fn(),
      modelStorage: vi.fn(),
      profiles: vi.fn(),
      runtimes: vi.fn(),
      system: vi.fn(),
    },
  };
});

function show() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <ModelsView initialRole="image" />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.mocked(api.catalog).mockReset().mockResolvedValue({
    items: [],
    next_cursor: null,
    stale: true,
  });
  vi.mocked(api.recipes).mockReset().mockResolvedValue([]);
  vi.mocked(api.models).mockReset().mockResolvedValue([]);
  vi.mocked(api.modelAssets).mockReset().mockResolvedValue([]);
  vi.mocked(api.jobs).mockReset().mockResolvedValue([]);
  vi.mocked(api.modelStorage).mockReset().mockResolvedValue({
    installed_count: 0,
    installed_bytes: 0,
    partial_download_count: 0,
    partial_download_bytes: 0,
    catalog_cache_bytes: 0,
  });
  vi.mocked(api.profiles).mockReset().mockResolvedValue([]);
  vi.mocked(api.runtimes).mockReset().mockResolvedValue([]);
  vi.mocked(api.system).mockReset().mockResolvedValue(
    {} as Awaited<ReturnType<typeof api.system>>,
  );
});

afterEach(() => {
  cleanup();
});

describe("stale model catalog source", () => {
  it.each([
    ["huggingface", "Hugging Face"],
    ["civitai", "CivitAI"],
  ])("names %s in the unavailable message", (source, provider) => {
    expect(catalogUnavailableMessage(source)).toBe(
      `Showing saved results while ${provider} is unavailable.`,
    );
  });

  it("updates the stale-results message when the selected source changes", async () => {
    show();
    expect(await screen.findByText(catalogUnavailableMessage("huggingface"))).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Model source"), { target: { value: "civitai" } });

    expect(await screen.findByText(catalogUnavailableMessage("civitai"))).toBeInTheDocument();
    await waitFor(() => expect(api.catalog).toHaveBeenLastCalledWith(
      "",
      "image",
      "trending",
      null,
      expect.anything(),
      "civitai",
    ));
  });
});
