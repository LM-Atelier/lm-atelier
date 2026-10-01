import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ModelsView } from "./ModelsView";
import type { CatalogModel, ModelInstall } from "./types";

vi.mock("./ModelUpdatesPanel", () => ({ ModelUpdatesPanel: () => null }));
vi.mock("./ModelUpdateProfileOffers", () => ({ ModelUpdateProfileOffers: () => null }));
vi.mock("./useCatalogInstall", () => ({ useCatalogInstall: () => ({
  pendingInstall: null, updateDownloads: [], dismissUpdate: vi.fn(), cancel: vi.fn(),
  prepare: { isPending: false, mutate: vi.fn() }, confirm: { isPending: false, mutate: vi.fn() },
}) }));
vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ...actual, api: { ...actual.api,
    models: vi.fn(), modelsPage: vi.fn(), profiles: vi.fn(), profilesPage: vi.fn(), catalogInstallMatches: vi.fn(),
    catalog: vi.fn(), recipes: vi.fn(), modelAssets: vi.fn(), jobs: vi.fn(), modelStorage: vi.fn(), runtimes: vi.fn(), system: vi.fn(),
  } };
});

const stamp = "2026-09-30T00:00:00Z";
function model(index: number): ModelInstall {
  return { id: `model-${index}`, source_id: null, name: `Installed ${index}`, role: "chat", engine: "mock",
    local_path: `neutral/${index}`, size_bytes: 100, compatibility: "likely", manifest_json: {}, active: true,
    readiness: "ready", capability_evidence: null, created_at: stamp, updated_at: stamp };
}
function card(template: string | null = null): CatalogModel {
  return { remote_id: "neutral/shared", name: template ?? "Off-page installation", author: "Neutral",
    pipeline_tag: null, library_name: null, downloads: 0, likes: 0, trending_score: null, last_modified: null,
    created_at: null, architecture: null, parameter_count: null, license_id: null,
    gated: false, private: false, tags: [], formats: [], quantizations: [], total_size_bytes: null,
    compatibility: "likely", compatibility_reasons: [], provider: "huggingface", workflow_template_id: template,
  };
}
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><ModelsView initialRole="image" /></QueryClientProvider>);
}

beforeEach(() => {
  vi.mocked(api.models).mockReset().mockResolvedValue([]);
  vi.mocked(api.profiles).mockReset().mockResolvedValue([]);
  vi.mocked(api.modelsPage).mockReset().mockResolvedValue([]);
  vi.mocked(api.profilesPage).mockReset().mockResolvedValue([]);
  vi.mocked(api.catalogInstallMatches).mockReset().mockResolvedValue({ remote_ids: [], workflow_template_ids: [] });
  vi.mocked(api.catalog).mockReset().mockResolvedValue({ items: [], next_cursor: null, stale: false });
  vi.mocked(api.recipes).mockReset().mockResolvedValue([]);
  vi.mocked(api.modelAssets).mockReset().mockResolvedValue([]);
  vi.mocked(api.jobs).mockReset().mockResolvedValue([]);
  vi.mocked(api.modelStorage).mockReset().mockResolvedValue({ installed_count: 51, installed_bytes: 5100, partial_download_count: 0, partial_download_bytes: 0, catalog_cache_bytes: 0 });
  vi.mocked(api.runtimes).mockReset().mockResolvedValue([]);
  vi.mocked(api.system).mockReset().mockResolvedValue({} as Awaited<ReturnType<typeof api.system>>);
});
afterEach(cleanup);

it("pages installed models and reads only each displayed model's first eligible profile", async () => {
  vi.mocked(api.modelsPage).mockImplementation(async ({ offset }) => offset ? [model(50)] : Array.from({ length: 50 }, (_, i) => model(i)));
  show();
  expect(await screen.findByText("Installed 0")).toBeInTheDocument();
  await waitFor(() => expect(api.profilesPage).toHaveBeenCalledWith({ limit: 1, installIds: ["model-0"], role: "chat", engine: "mock" }));
  expect(api.modelsPage).toHaveBeenCalledWith(expect.objectContaining({ limit: 50, offset: 0 }));
  expect(api.models).not.toHaveBeenCalled();
  expect(api.profiles).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "More installed models" }));
  expect(await screen.findByText("Installed 50")).toBeInTheDocument();
  expect(screen.getByText("Installed 0")).toBeInTheDocument();
  expect(api.modelsPage).toHaveBeenLastCalledWith(expect.objectContaining({ limit: 50, offset: 50 }));
});

it("sends installed search and capability filters before loading matching pages", async () => {
  show();
  fireEvent.change(screen.getByRole("textbox", { name: "Search installed models" }), { target: { value: "Straße %_" } });
  fireEvent.change(screen.getByLabelText("Installed chat capability"), { target: { value: "vision" } });
  await waitFor(() => expect(api.modelsPage).toHaveBeenLastCalledWith({ limit: 50, offset: 0, search: "Straße %_", chatCapability: "vision" }));
  expect(await screen.findByText("No installed models match.")).toBeInTheDocument();
});

it("keeps exact installed catalog badges outside the visible model page and separates variants", async () => {
  vi.mocked(api.catalog).mockResolvedValue({ items: [card(), card("neutral-edit"), card("neutral-create")], next_cursor: null, stale: false });
  vi.mocked(api.catalogInstallMatches).mockResolvedValue({ remote_ids: ["neutral/shared"], workflow_template_ids: ["neutral-edit"] });
  show();
  const outside = await screen.findByRole("heading", { name: "Off-page installation" });
  await waitFor(() => expect(within(outside.closest("article")!).getByRole("button", { name: "Installed" })).toBeDisabled());
  expect(within(screen.getByRole("heading", { name: "neutral-edit" }).closest("article")!).getByRole("button", { name: "Installed" })).toBeDisabled();
  expect(within(screen.getByRole("heading", { name: "neutral-create" }).closest("article")!).getByRole("button", { name: "Install" })).toBeEnabled();
});

it("keeps catalog actions unavailable while status fails and retries the lookup", async () => {
  vi.mocked(api.catalog).mockResolvedValue({ items: [card()], next_cursor: null, stale: false });
  vi.mocked(api.catalogInstallMatches).mockRejectedValueOnce(new Error("Status read failed"));
  show();
  expect(await screen.findByRole("button", { name: "Status unavailable" })).toBeDisabled();
  expect(screen.queryByRole("button", { name: "Install" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry installation status" }));
  expect(await screen.findByRole("button", { name: "Install" })).toBeEnabled();
});

it("does not offer to recreate a profile when its exact read fails", async () => {
  vi.mocked(api.modelsPage).mockResolvedValue([model(1)]);
  vi.mocked(api.profilesPage).mockRejectedValueOnce(new Error("Profile read failed"));
  show();
  expect(await screen.findByRole("button", { name: "Retry profile for Installed 1" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Add Installed 1 to model selectors" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry profile for Installed 1" }));
  expect(await screen.findByRole("button", { name: "Add Installed 1 to model selectors" })).toBeEnabled();
});

it("keeps installed rows when the next page fails and retries that page", async () => {
  vi.mocked(api.modelsPage).mockResolvedValueOnce(Array.from({ length: 50 }, (_, i) => model(i)))
    .mockRejectedValueOnce(new Error("Next page failed")).mockResolvedValueOnce([model(50)]);
  show();
  fireEvent.click(await screen.findByRole("button", { name: "More installed models" }));
  expect(await screen.findByText("Next page failed")).toBeInTheDocument();
  expect(screen.getByText("Installed 0")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry installed models" }));
  expect(await screen.findByText("Installed 50")).toBeInTheDocument();
  expect(api.modelsPage).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 50 }));
});


it("bounds catalog status batches when more than two hundred cards are visible", async () => {
  const items = Array.from({ length: 201 }, (_, index) => ({ ...card(), name: `Catalog ${index}`, remote_id: `neutral/${index}`, workflow_template_id: `template-${index}` }));
  vi.mocked(api.catalog).mockResolvedValue({ items, next_cursor: null, stale: false });
  vi.mocked(api.catalogInstallMatches).mockImplementation(async (options) => ({ remote_ids: [], workflow_template_ids: options.workflowTemplateIds.includes("template-200") ? ["template-200"] : [] }));
  show();
  expect(await screen.findByRole("button", { name: "Installed" })).toBeDisabled();
  expect(api.catalogInstallMatches).toHaveBeenCalledTimes(2);
  for (const [options] of vi.mocked(api.catalogInstallMatches).mock.calls) {
    expect(options.remoteIds.length).toBeLessThanOrEqual(200);
    expect(options.workflowTemplateIds.length).toBeLessThanOrEqual(200);
  }
  expect(api.catalogInstallMatches).toHaveBeenLastCalledWith({ role: "image", remoteIds: ["neutral/200"], workflowTemplateIds: ["template-200"] });
});

it("holds catalog and profile actions until their identity reads complete", async () => {
  let finish!: (matches: Awaited<ReturnType<typeof api.catalogInstallMatches>>) => void;
  vi.mocked(api.catalogInstallMatches).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  vi.mocked(api.catalog).mockResolvedValue({ items: [card()], next_cursor: null, stale: false });
  vi.mocked(api.modelsPage).mockResolvedValue([model(1)]);
  vi.mocked(api.profilesPage).mockReturnValue(new Promise(() => {}));
  show();
  expect(await screen.findByRole("button", { name: "Checking installation…" })).toBeDisabled();
  expect(await screen.findByText("Installed 1")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Add Installed 1 to model selectors" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Install" })).not.toBeInTheDocument();
  await act(async () => { finish({ remote_ids: ["neutral/shared"], workflow_template_ids: [] }); });
  expect(await screen.findByRole("button", { name: "Installed" })).toBeDisabled();
});
