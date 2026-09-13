/** Suggested LoRAs for the model a workflow runs, installed only on request. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { LoraSuggestions } from "./LoraSuggestions";
import type { CatalogModel, CatalogPreflight, LoraSuggestions as Suggestions } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: { workflowLoraSuggestions: vi.fn(), catalogPreflight: vi.fn(), download: vi.fn() },
  };
});

function card(remoteId: string, overrides: Partial<CatalogModel> = {}): CatalogModel {
  return {
    provider: "civitai",
    remote_id: remoteId,
    name: `Version ${remoteId}`,
    author: "garden-studio",
    pipeline_tag: null,
    tags: [],
    downloads: 12000,
    likes: 850,
    trending_score: null,
    created_at: null,
    last_modified: null,
    gated: null,
    private: false,
    library_name: null,
    architecture: "SDXL 1.0",
    formats: ["safetensors"],
    quantizations: [],
    parameter_count: null,
    license_id: null,
    total_size_bytes: 1024,
    compatibility: "advanced_import",
    compatibility_reasons: [],
    parent_model_id: `model-${remoteId}`,
    parent_model_name: `Watercolor wash ${remoteId}`,
    content_rating: "general",
    ...overrides,
  };
}

function answer(overrides: Partial<Suggestions> = {}): Suggestions {
  return { family: "stable-diffusion-xl", gap: null, items: [card("1")], next_cursor: null, stale: false, ...overrides };
}

const PREFLIGHT: CatalogPreflight = {
  remote_id: "1",
  source_remote_id: null,
  revision: "1",
  selected_files: ["watercolor.safetensors"],
  expected_sha256: { "watercolor.safetensors": "a".repeat(64) },
  comfy_paths: { loras: "." },
  workflow_template_id: null,
  workflow_template_sha256: null,
  download_bytes: 1024,
  available_disk_bytes: 10 ** 9,
  estimated_ram_bytes: null,
  estimated_vram_bytes: null,
  can_install: true,
  auxiliary_kind: "lora",
  content_rating: "general",
  install_plan: { id: "plan-1", plan_hash: "b".repeat(64), compatibility: "supported", family: null, failure_code: null, failure_reason: null },
  checks: [],
};

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><LoraSuggestions revisionId="wfrev-garden" /></QueryClientProvider>);
}

function openSuggestions() {
  const details = screen.getByText("Top-rated LoRAs for this model").closest("details")!;
  details.open = true;
  fireEvent(details, new Event("toggle"));
}

beforeEach(() => {
  vi.mocked(api.workflowLoraSuggestions).mockReset();
  vi.mocked(api.catalogPreflight).mockReset();
  vi.mocked(api.download).mockReset();
});

afterEach(() => {
  cleanup();
});

it("asks for nothing until opened, then lists each LoRA with its maker and how liked it is", async () => {
  vi.mocked(api.workflowLoraSuggestions).mockResolvedValue(answer());
  show();
  expect(api.workflowLoraSuggestions).not.toHaveBeenCalled();

  openSuggestions();

  expect(await screen.findByText("Watercolor wash 1")).toBeInTheDocument();
  expect(screen.getByText(`garden-studio · ${(850).toLocaleString()} likes · ${(12000).toLocaleString()} downloads`)).toBeInTheDocument();
  expect(api.workflowLoraSuggestions).toHaveBeenCalledWith("wfrev-garden", expect.anything());
});

it("says why there are no suggestions, and when saved ones are shown", async () => {
  vi.mocked(api.workflowLoraSuggestions).mockResolvedValue(answer({ gap: "family_unknown", items: [], family: null }));
  show();
  openSuggestions();
  expect(await screen.findByText("The model this workflow runs could not be identified, so there are no suggestions for it.")).toBeInTheDocument();
  cleanup();

  vi.mocked(api.workflowLoraSuggestions).mockResolvedValue(answer({ gap: "family_unsupported", items: [] }));
  show();
  openSuggestions();
  expect(await screen.findByText("There are no LoRA suggestions for this kind of model yet.")).toBeInTheDocument();
  cleanup();

  vi.mocked(api.workflowLoraSuggestions).mockResolvedValue(answer({ stale: true, items: [] }));
  show();
  openSuggestions();
  expect(await screen.findByText("Showing saved suggestions while CivitAI is unavailable.")).toBeInTheDocument();
  expect(screen.getByText("No suggestions that are not already installed.")).toBeInTheDocument();
});

it("installs a suggestion as a LoRA only after the download is confirmed", async () => {
  vi.mocked(api.workflowLoraSuggestions).mockResolvedValue(answer());
  vi.mocked(api.catalogPreflight).mockResolvedValue(PREFLIGHT);
  vi.mocked(api.download).mockResolvedValue({} as Awaited<ReturnType<typeof api.download>>);
  show();
  openSuggestions();

  fireEvent.click(await screen.findByRole("button", { name: "Install Watercolor wash 1" }));

  expect(await screen.findByRole("dialog")).toBeInTheDocument();
  expect(api.catalogPreflight).toHaveBeenCalledWith("1", "image", "comfyui", "1", [], "lora", null, "civitai");
  expect(api.download).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole("button", { name: /^Download / }));

  await waitFor(() => expect(api.download).toHaveBeenCalledWith(
    "1", null, "image", "comfyui", "1", ["watercolor.safetensors"], { "watercolor.safetensors": "a".repeat(64) },
    {}, { loras: "." }, null, null, "plan-1", "lora", "general",
  ));
  expect(await screen.findByText("Installing. It can be added once the download finishes.")).toBeInTheDocument();
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("shows why a suggestion cannot be installed and starts nothing", async () => {
  vi.mocked(api.workflowLoraSuggestions).mockResolvedValue(answer());
  vi.mocked(api.catalogPreflight).mockResolvedValue({
    ...PREFLIGHT,
    can_install: false,
    checks: [{ id: "scan", label: "Scan", status: "block", detail: "This file has not passed its safety scan." }],
  });
  show();
  openSuggestions();

  fireEvent.click(await screen.findByRole("button", { name: "Install Watercolor wash 1" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("This file has not passed its safety scan.");
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(api.download).not.toHaveBeenCalled();
});
