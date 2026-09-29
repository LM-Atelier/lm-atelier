import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
    catalog: vi.fn(), recipes: vi.fn(), models: vi.fn(), modelAssets: vi.fn(),
    updateModelAsset: vi.fn(), jobs: vi.fn(), modelStorage: vi.fn(), profiles: vi.fn(),
    runtimes: vi.fn(), system: vi.fn(),
  } };
});

const stamp = "2026-09-29T00:00:00Z";
const asset = {
  id: "watercolor", name: "Watercolor", source_id: null, kind: "lora" as const,
  family: "sdxl", size_bytes: 1024, manifest_json: {}, active: true,
  use_case: "Watercolor landscapes", use_case_derived: true, auto_apply: false,
  default_model_strength: 1, default_clip_strength: 1, typed_trigger_words: [],
  verified_at: stamp, created_at: stamp, updated_at: stamp,
};
beforeEach(() => {
  vi.mocked(api.catalog).mockResolvedValue({ items: [], next_cursor: null, stale: false });
  vi.mocked(api.recipes).mockResolvedValue([]);
  vi.mocked(api.models).mockResolvedValue([]);
  vi.mocked(api.modelAssets).mockResolvedValue([asset]);
  vi.mocked(api.jobs).mockResolvedValue([]);
  vi.mocked(api.profiles).mockResolvedValue([]);
  vi.mocked(api.runtimes).mockResolvedValue([]);
  vi.mocked(api.system).mockResolvedValue({} as Awaited<ReturnType<typeof api.system>>);
  vi.mocked(api.modelStorage).mockResolvedValue({ installed_count: 0, installed_bytes: 0, partial_download_count: 0, partial_download_bytes: 0, catalog_cache_bytes: 0 });
  vi.mocked(api.updateModelAsset).mockReset().mockResolvedValue(asset);
});
afterEach(cleanup);

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><ModelsView initialRole="image" /></QueryClientProvider>);
}

it("shows derived text without automatic use and keeps its provenance on strength edits", async () => {
  let finishSave!: (value: typeof asset) => void;
  vi.mocked(api.updateModelAsset).mockImplementation(() => new Promise((resolve) => { finishSave = resolve; }));
  show();
  expect(await screen.findByText("Derived · Watercolor landscapes")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Edit Auto rules" }));
  expect(screen.getByLabelText("Auto use case for Watercolor")).toHaveFocus();
  expect(screen.getByText("Derived from provider metadata. You can edit it.")).toBeInTheDocument();
  expect(screen.getByLabelText("Use Watercolor automatically")).not.toBeChecked();
  fireEvent.change(screen.getByLabelText("Default model strength for Watercolor"), { target: { value: "0.5" } });
  const save = screen.getByRole("button", { name: "Save" });
  save.focus();
  fireEvent.click(save);
  await waitFor(() => expect(api.updateModelAsset).toHaveBeenCalledWith("watercolor", {
    auto_apply: false, default_model_strength: 0.5, default_clip_strength: 1,
  }));
  await screen.findByRole("button", { name: "Saving…" });
  expect(save).toHaveFocus();
  expect(save).toHaveAttribute("aria-disabled", "true");
  await act(async () => { fireEvent.click(save); });
  expect(api.updateModelAsset).toHaveBeenCalledTimes(1);
  finishSave(asset);
  await waitFor(() => expect(screen.getByRole("button", { name: "Edit Auto rules" })).toHaveFocus());
});

it("retains an explicit clear through a failed save and restores the label when cancelled", async () => {
  vi.mocked(api.updateModelAsset).mockRejectedValueOnce(new Error("Save failed"));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Edit Auto rules" }));
  const field = screen.getByLabelText("Auto use case for Watercolor");
  fireEvent.change(field, { target: { value: "" } });
  expect(screen.queryByText("Derived from provider metadata. You can edit it.")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText("Save failed");
  expect(field).toHaveValue("");
  expect(api.updateModelAsset).toHaveBeenCalledWith("watercolor", expect.objectContaining({ use_case: "" }));
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.getByRole("button", { name: "Edit Auto rules" })).toHaveFocus();
  fireEvent.click(screen.getByRole("button", { name: "Edit Auto rules" }));
  expect(screen.getByLabelText("Auto use case for Watercolor")).toHaveValue("Watercolor landscapes");
  expect(screen.getByText("Derived from provider metadata. You can edit it.")).toBeInTheDocument();
});

it("does not label a manual description as derived", async () => {
  vi.mocked(api.modelAssets).mockResolvedValue([{ ...asset, use_case_derived: false }]);
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Edit Auto rules" }));
  expect(screen.queryByText(/Derived/)).not.toBeInTheDocument();
});

it("restores the trigger-word editor launch button after save and cancel", async () => {
  show();
  const launch = await screen.findByRole("button", { name: "Edit trigger words" });
  fireEvent.click(launch);
  const field = screen.getByLabelText("Trigger words for Watercolor");
  expect(field).toHaveFocus();
  fireEvent.change(field, { target: { value: "ink wash" } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(launch).toHaveFocus());
  expect(api.updateModelAsset).toHaveBeenCalledWith("watercolor", { typed_trigger_words: ["ink wash"] });
  fireEvent.click(launch);
  expect(screen.getByLabelText("Trigger words for Watercolor")).toHaveFocus();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(launch).toHaveFocus();
  expect(api.updateModelAsset).toHaveBeenCalledTimes(1);
});
