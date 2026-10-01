import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ModelsView } from "./ModelsView";
import type { ModelInstall, ModelProfile } from "./types";

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
    catalog: vi.fn(), recipes: vi.fn(), models: vi.fn(), modelsPage: vi.fn(() => api.models()), catalogInstallMatches: vi.fn(async () => ({ remote_ids: [], workflow_template_ids: [] })), modelAssets: vi.fn(), jobs: vi.fn(),
    modelStorage: vi.fn(), profiles: vi.fn(), profilesPage: vi.fn(async (options) => (await api.profiles()).filter((profile) => !options.installIds || options.installIds.includes(profile.model_install_id ?? "")).slice(0, options.limit)), runtimes: vi.fn(), system: vi.fn(),
    updateProfile: vi.fn(),
  } };
});

const stamp = "2026-09-30T00:00:00Z";
function profile(id: string): ModelProfile {
  return {
    id, model_install_id: id, name: id, use_case: `${id} scenes`, role: "image",
    engine: "comfyui", use_case_derived: false, load_settings_json: {},
    request_settings_json: {}, is_default: false,
  };
}
function model(id: string): ModelInstall {
  return {
    id, name: id, role: "image", engine: "comfyui", active: true, source_id: null,
    local_path: "neutral", compatibility: "likely", capability_evidence: null,
    created_at: stamp, updated_at: stamp, readiness: "ready", size_bytes: 1024,
    manifest_json: {},
  };
}
function deferred() {
  let resolve!: (value: ModelProfile) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<ModelProfile>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
beforeEach(() => {
  vi.mocked(api.catalog).mockResolvedValue({ items: [], next_cursor: null, stale: false });
  vi.mocked(api.recipes).mockResolvedValue([]);
  vi.mocked(api.models).mockResolvedValue([model("Watercolor"), model("Ink")]);
  vi.mocked(api.modelAssets).mockResolvedValue([]);
  vi.mocked(api.jobs).mockResolvedValue([]);
  vi.mocked(api.profiles).mockResolvedValue([profile("Watercolor"), profile("Ink")]);
  vi.mocked(api.runtimes).mockResolvedValue([]);
  vi.mocked(api.system).mockResolvedValue({} as Awaited<ReturnType<typeof api.system>>);
  vi.mocked(api.modelStorage).mockResolvedValue({ installed_count: 2, installed_bytes: 2048, partial_download_count: 0, partial_download_bytes: 0, catalog_cache_bytes: 0 });
  vi.mocked(api.updateProfile).mockReset().mockResolvedValue(profile("Watercolor"));
});
afterEach(cleanup);
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><ModelsView initialRole="image" /></QueryClientProvider>);
}
async function open(name = "Watercolor") {
  const launch = await screen.findByRole("button", { name: `Edit use case for ${name}` });
  launch.focus();
  fireEvent.click(launch);
  return { launch, field: screen.getByLabelText(`Best uses for ${name}`) };
}
function formOf(field: HTMLElement): HTMLFormElement {
  const form = field.closest("form");
  if (!form) throw new Error("Missing use-case form");
  return form;
}

it("focuses the selected model editor and returns to its trigger after cancel", async () => {
  show();
  const { launch, field } = await open("Ink");
  expect(field).toHaveFocus();
  fireEvent.change(field, { target: { value: "Unsaved draft" } });
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(launch).toHaveFocus();
  fireEvent.click(launch);
  expect(screen.getByLabelText("Best uses for Ink")).toHaveValue("Ink scenes");
  expect(api.updateProfile).not.toHaveBeenCalled();
});

it("keeps pending actions focusable and refuses repeated submit or cancel", async () => {
  const held = deferred();
  vi.mocked(api.updateProfile).mockReturnValue(held.promise);
  show();
  const { launch, field } = await open();
  fireEvent.change(field, { target: { value: "  Landscapes  " } });
  const save = screen.getByRole("button", { name: "Save" });
  save.focus();
  fireEvent.click(save);
  await screen.findByRole("button", { name: "Saving…" });
  expect(save).not.toBeDisabled();
  expect(save).toHaveAttribute("aria-disabled", "true");
  expect(save).toHaveFocus();
  const cancel = screen.getByRole("button", { name: "Cancel" });
  expect(cancel).not.toBeDisabled();
  expect(cancel).toHaveAttribute("aria-disabled", "true");
  await act(async () => { fireEvent.click(save); fireEvent.submit(formOf(field)); fireEvent.click(cancel); });
  expect(field).toBeInTheDocument();
  expect(api.updateProfile).toHaveBeenCalledTimes(1);
  expect(api.updateProfile).toHaveBeenCalledWith("Watercolor", { use_case: "Landscapes" });
  await act(async () => { held.resolve({ ...profile("Watercolor"), use_case: "Landscapes" }); });
  await waitFor(() => expect(launch).toHaveFocus());
});

it("retains a failed edit and restores focus after a successful retry", async () => {
  vi.mocked(api.updateProfile).mockRejectedValueOnce(new Error("Save failed"));
  show();
  const { launch, field } = await open();
  fireEvent.change(field, { target: { value: "" } });
  const save = screen.getByRole("button", { name: "Save" });
  save.focus();
  fireEvent.click(save);
  await screen.findByText("Save failed");
  expect(field).toHaveValue("");
  expect(save).toHaveFocus();
  expect(save).toHaveAttribute("aria-disabled", "false");
  fireEvent.click(save);
  await waitFor(() => expect(launch).toHaveFocus());
  expect(api.updateProfile).toHaveBeenCalledTimes(2);
  expect(api.updateProfile).toHaveBeenLastCalledWith("Watercolor", { use_case: "" });
});

it("refuses an unchanged keyboard submission without closing the editor", async () => {
  show();
  const { field } = await open();
  fireEvent.change(field, { target: { value: "  Watercolor scenes  " } });
  const save = screen.getByRole("button", { name: "Save" });
  await act(async () => { fireEvent.submit(formOf(field)); });
  expect(api.updateProfile).not.toHaveBeenCalled();
  expect(save).not.toBeDisabled();
  expect(save).toHaveAttribute("aria-disabled", "true");
  expect(field).toBeInTheDocument();
});

it("keeps each pending save scoped to its model and does not steal sibling focus", async () => {
  const first = deferred();
  const second = deferred();
  vi.mocked(api.updateProfile).mockImplementation((id) => id === "Watercolor" ? first.promise : second.promise);
  show();
  const firstEditor = await open();
  fireEvent.change(firstEditor.field, { target: { value: "New watercolor" } });
  fireEvent.submit(formOf(firstEditor.field));
  await waitFor(() => expect(api.updateProfile).toHaveBeenCalledTimes(1));
  const secondEditor = await open("Ink");
  fireEvent.change(secondEditor.field, { target: { value: "New ink" } });
  fireEvent.submit(formOf(secondEditor.field));
  await waitFor(() => expect(api.updateProfile).toHaveBeenCalledTimes(2));
  await act(async () => { fireEvent.submit(formOf(firstEditor.field)); });
  expect(api.updateProfile).toHaveBeenCalledTimes(2);
  secondEditor.field.focus();
  await act(async () => { first.resolve({ ...profile("Watercolor"), use_case: "New watercolor" }); });
  await waitFor(() => expect(firstEditor.field).not.toBeInTheDocument());
  expect(secondEditor.field).toHaveFocus();
  await act(async () => { second.resolve({ ...profile("Ink"), use_case: "New ink" }); });
  await waitFor(() => expect(secondEditor.launch).toHaveFocus());
});
