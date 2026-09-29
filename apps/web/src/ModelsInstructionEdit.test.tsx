import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ModelsView } from "./ModelsView";
import type { ModelAssetInstall, ModelInstall } from "./types";

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
    catalog: vi.fn(), recipes: vi.fn(), models: vi.fn(), modelAssets: vi.fn(),
    jobs: vi.fn(), modelStorage: vi.fn(), profiles: vi.fn(), runtimes: vi.fn(), system: vi.fn(),
  } };
});

const stamp = "2026-09-29T00:00:00Z";
const model: ModelInstall = {
  id: "painter", name: "Primary painter", source_id: null, role: "image", engine: "comfyui",
  local_path: "C:/neutral/painter", size_bytes: 1024, compatibility: "likely",
  manifest_json: {}, active: true, readiness: "ready", capability_evidence: null,
  created_at: stamp, updated_at: stamp,
};
const asset: ModelAssetInstall = {
  id: "adapter", name: "Edit adapter", source_id: null, kind: "lora",
  family: "sdxl", size_bytes: 1024, manifest_json: {}, active: true,
  use_case: "", use_case_derived: false, auto_apply: false,
  default_model_strength: 1, default_clip_strength: 1, typed_trigger_words: [],
  verified_at: stamp, created_at: stamp, updated_at: stamp,
};
beforeEach(() => {
  vi.mocked(api.catalog).mockResolvedValue({ items: [], next_cursor: null, stale: false });
  vi.mocked(api.recipes).mockResolvedValue([]);
  vi.mocked(api.models).mockResolvedValue([model]);
  vi.mocked(api.modelAssets).mockResolvedValue([asset]);
  vi.mocked(api.jobs).mockResolvedValue([]);
  vi.mocked(api.profiles).mockResolvedValue([]);
  vi.mocked(api.runtimes).mockResolvedValue([]);
  vi.mocked(api.system).mockResolvedValue({} as Awaited<ReturnType<typeof api.system>>);
  vi.mocked(api.modelStorage).mockResolvedValue({ installed_count: 1, installed_bytes: 1024, partial_download_count: 0, partial_download_bytes: 0, catalog_cache_bytes: 0 });
});
afterEach(cleanup);

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><ModelsView initialRole="image" /></QueryClientProvider>);
}

it("shows source declarations separately from runtime readiness for models and assets", async () => {
  const manifest_json = { instruction_edit_capability: "declared" };
  vi.mocked(api.models).mockResolvedValue([{ ...model, manifest_json }]);
  vi.mocked(api.modelAssets).mockResolvedValue([{ ...asset, manifest_json }]);
  show();
  await screen.findByText("Primary painter");
  await screen.findByText("Edit adapter");
  expect(screen.getAllByText("Instruction editing: declared by source")).toHaveLength(2);
  expect(screen.getAllByText("Requires a compatible workflow.")).toHaveLength(2);
  expect(screen.getByText("Runtime verified")).toBeInTheDocument();
});

it.each([undefined, "unknown", true, ["declared"], { declared: true }])(
  "keeps missing or malformed declaration %j unknown despite runtime readiness",
  async (value) => {
    vi.mocked(api.models).mockResolvedValue([{ ...model, manifest_json: { instruction_edit_capability: value } }]);
    show();
    const name = await screen.findByText("Primary painter");
    const copy = within(name.parentElement!);
    expect(copy.getByText("Instruction editing: unknown")).toBeInTheDocument();
    expect(copy.getByText("Runtime verified")).toBeInTheDocument();
    expect(copy.queryByText("Instruction editing: declared by source")).not.toBeInTheDocument();
  },
);

it("does not present an unrelated asset as an instruction editor", async () => {
  vi.mocked(api.modelAssets).mockResolvedValue([{ ...asset, kind: "upscaler", manifest_json: { instruction_edit_capability: "declared" } }]);
  show();
  const name = await screen.findByText("Edit adapter");
  expect(within(name.parentElement!).queryByText(/Instruction editing:/)).not.toBeInTheDocument();
});
