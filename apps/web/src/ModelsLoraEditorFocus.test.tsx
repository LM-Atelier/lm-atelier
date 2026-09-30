import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
const assets = [asset, { ...asset, id: "atelier-ink", name: "Atelier Ink" }];
beforeEach(() => {
  vi.mocked(api.catalog).mockResolvedValue({ items: [], next_cursor: null, stale: false });
  vi.mocked(api.recipes).mockResolvedValue([]);
  vi.mocked(api.models).mockResolvedValue([]);
  vi.mocked(api.modelAssets).mockResolvedValue(assets);
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


function row(name: string) {
  return screen.getByText(name, { exact: true }).closest("div")!;
}

async function openEditor(name: string, kind: "rules" | "words") {
  await screen.findByText(name, { exact: true });
  fireEvent.click(within(row(name)).getByRole("button", {
    name: kind === "rules" ? "Edit Auto rules" : "Edit trigger words",
  }));
  const field = screen.getByLabelText(`${kind === "rules" ? "Auto use case" : "Trigger words"} for ${name}`);
  fireEvent.change(field, { target: { value: "Ink on paper" } });
  return { field, form: field.closest("form")!, save: within(field.closest("form")!).getByRole("button", { name: "Save" }) };
}

it.each(["rules", "words"] as const)("keeps a pending %s save guarded when another asset starts saving", async (kind) => {
  const finishes: (() => void)[] = [];
  vi.mocked(api.updateModelAsset).mockImplementation((id) => new Promise((resolve) => {
    finishes.push(() => resolve(assets.find((item) => item.id === id)!));
  }));
  show();
  const first = await openEditor("Watercolor", kind);
  fireEvent.click(first.save);
  await waitFor(() => expect(api.updateModelAsset).toHaveBeenCalledTimes(1));
  const second = await openEditor("Atelier Ink", kind);
  fireEvent.click(second.save);
  await waitFor(() => expect(api.updateModelAsset).toHaveBeenCalledTimes(2));
  try {
    await act(async () => { fireEvent.submit(first.form); });
    expect(api.updateModelAsset).toHaveBeenCalledTimes(2);
    expect(first.save).toHaveAttribute("aria-disabled", "true");
  } finally {
    await act(async () => { finishes.forEach((finish) => finish()); });
  }
});

it.each(["rules", "words"] as const)("does not take focus from another editor when a %s save completes", async (kind) => {
  let finish!: () => void;
  vi.mocked(api.updateModelAsset).mockImplementation(() => new Promise((resolve) => { finish = () => resolve(asset); }));
  show();
  const first = await openEditor("Watercolor", kind);
  fireEvent.click(first.save);
  await waitFor(() => expect(api.updateModelAsset).toHaveBeenCalledTimes(1));
  const second = await openEditor("Atelier Ink", "rules");
  second.field.focus();
  expect(second.field).toHaveFocus();
  await act(async () => { finish(); });
  await waitFor(() => expect(first.field).not.toBeInTheDocument());
  expect(second.field).toHaveFocus();
});

it("keeps both drafts fixed while their asset is saving", async () => {
  let finish!: () => void;
  vi.mocked(api.updateModelAsset).mockImplementation(() => new Promise((resolve) => { finish = () => resolve(asset); }));
  show();
  const rules = await openEditor("Watercolor", "rules");
  const words = await openEditor("Watercolor", "words");
  fireEvent.click(rules.save);
  await waitFor(() => expect(api.updateModelAsset).toHaveBeenCalledTimes(1));
  try {
    expect(rules.field).toHaveAttribute("readonly");
    expect(words.field).toHaveAttribute("readonly");
    for (const label of ["Base model", "Default model strength", "Default CLIP strength"]) {
      expect(screen.getByLabelText(`${label} for Watercolor`)).toHaveAttribute("readonly");
    }
    const auto = screen.getByLabelText("Use Watercolor automatically");
    expect(auto).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(auto);
    expect(auto).not.toBeChecked();
    fireEvent.click(within(words.form).getByRole("button", { name: "Cancel" }));
    expect(words.field).toBeInTheDocument();
    expect(rules.save).toHaveAttribute("aria-disabled", "true");
  } finally {
    await act(async () => { finish(); });
  }
});

it("retains the draft and releases its controls after a failed update", async () => {
  vi.mocked(api.updateModelAsset).mockRejectedValueOnce(new Error("Save failed"));
  show();
  const editor = await openEditor("Watercolor", "rules");
  fireEvent.click(editor.save);
  await screen.findByText("Save failed");
  expect(editor.field).toHaveValue("Ink on paper");
  expect(editor.field).not.toHaveAttribute("readonly");
  expect(editor.save).toHaveAttribute("aria-disabled", "false");
  editor.field.focus();
  fireEvent.click(within(editor.form).getByRole("button", { name: "Cancel" }));
  expect(within(row("Watercolor")).getByRole("button", { name: "Edit Auto rules" })).toHaveFocus();
});
