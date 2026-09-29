import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ModelsView } from "./ModelsView";

vi.mock("./ModelUpdatesPanel", () => ({ ModelUpdatesPanel: () => null }));
vi.mock("./useCatalogInstall", () => ({
  useCatalogInstall: () => ({
    pendingInstall: null, updateDownloads: [], dismissUpdate: vi.fn(), cancel: vi.fn(),
    prepare: { isPending: false, variables: undefined, mutate: vi.fn() },
    confirm: { isPending: false, mutate: vi.fn() },
  }),
}));
vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ...actual, api: { ...actual.api,
    catalog: vi.fn(), recipes: vi.fn(), models: vi.fn(), modelAssets: vi.fn(), jobs: vi.fn(),
    modelStorage: vi.fn(), profiles: vi.fn(), runtimes: vi.fn(), system: vi.fn(),
    suggestProfileUseCase: vi.fn(), suggestLoraUseCase: vi.fn(),
    updateProfile: vi.fn(), updateModelAsset: vi.fn(),
  } };
});

const stamp = "2026-09-29T00:00:00Z";
beforeEach(() => {
  vi.mocked(api.catalog).mockResolvedValue({ items: [], next_cursor: null, stale: false });
  vi.mocked(api.recipes).mockResolvedValue([]);
  vi.mocked(api.jobs).mockResolvedValue([]);
  vi.mocked(api.runtimes).mockResolvedValue([]);
  vi.mocked(api.system).mockResolvedValue({} as Awaited<ReturnType<typeof api.system>>);
  vi.mocked(api.modelStorage).mockResolvedValue({ installed_count: 0, installed_bytes: 0, partial_download_count: 0, partial_download_bytes: 0, catalog_cache_bytes: 0 });
  vi.mocked(api.models).mockResolvedValue([{
    id: "model", name: "Landscape model", role: "image", engine: "comfyui", active: true,
    source_id: null, local_path: "neutral", compatibility: "likely", capability_evidence: null,
    created_at: stamp, updated_at: stamp,
    readiness: "ready", size_bytes: 1024, manifest_json: { provider_description: "Constructed landscape description" },
  }]);
  vi.mocked(api.profiles).mockResolvedValue([{
    id: "profile", model_install_id: "model", name: "Landscape profile", use_case: "Existing model use case",
    role: "image", engine: "comfyui", use_case_derived: false, load_settings_json: {}, request_settings_json: {}, is_default: false,
  }]);
  vi.mocked(api.modelAssets).mockResolvedValue([{
    id: "lora", name: "Watercolor LoRA", kind: "lora", active: true, size_bytes: 1024,
    source_id: null, family: "sdxl", verified_at: stamp, created_at: stamp, updated_at: stamp,
    use_case: "Existing LoRA use case", use_case_derived: false, auto_apply: false,
    default_model_strength: 1, default_clip_strength: 1, typed_trigger_words: [],
    manifest_json: { provider_description: "Constructed watercolor description" },
  }]);
  vi.mocked(api.suggestProfileUseCase).mockResolvedValue({ suggestion: "Landscape scenes." });
  vi.mocked(api.suggestLoraUseCase).mockResolvedValue({ suggestion: "Watercolor scenes." });
});
afterEach(cleanup);

it.each([
  ["Landscape model", "profile", "Existing model use case", "Landscape scenes."],
  ["Watercolor LoRA", "lora", "Existing LoRA use case", "Watercolor scenes."],
])("offers a confirmed suggestion from the installed %s row", async (name, id, expected, suggested) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><ModelsView initialRole="image" /></QueryClientProvider>);
  const opener = await screen.findByRole("button", { name: `Suggest use case for ${name}` });
  opener.focus();
  fireEvent.click(opener);
  expect(await screen.findByRole("textbox", { name: `Suggested use case for ${name}` })).toHaveValue(suggested);
  const request = id === "profile" ? api.suggestProfileUseCase : api.suggestLoraUseCase;
  expect(request).toHaveBeenCalledWith(id, expected, expect.any(AbortSignal));
  expect(api.updateProfile).not.toHaveBeenCalled();
  expect(api.updateModelAsset).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(opener).toHaveFocus();
});
