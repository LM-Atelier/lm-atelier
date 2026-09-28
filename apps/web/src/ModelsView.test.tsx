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
    updateDownloads: [],
    dismissUpdate: vi.fn(),
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
      updateModelAsset: vi.fn(),
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

describe("trigger words on an installed LoRA", () => {
  it("shows the file's words and the typed ones apart, and records new ones", async () => {
    const stamp = "2026-09-23T00:00:00Z";
    const asset = {
      id: "asset-ink",
      source_id: null,
      name: "Atelier Ink",
      kind: "lora" as const,
      family: "sdxl",
      size_bytes: 1024,
      manifest_json: { metadata: { trigger_words: ["ink wash"] } },
      active: true,
      use_case: "",
      auto_apply: false,
      default_model_strength: 1,
      default_clip_strength: 1,
      typed_trigger_words: ["studio glow"],
      verified_at: stamp,
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.modelAssets).mockResolvedValue([asset]);
    vi.mocked(api.updateModelAsset).mockReset().mockResolvedValue({
      ...asset,
      typed_trigger_words: ["studio glow", "soft edge"],
    });
    show();

    expect(await screen.findByText("From the file: ink wash · Yours: studio glow")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Edit trigger words" }));
    // The open editor lists the file's words in full, in place of the row's glance.
    expect(screen.queryByText("From the file: ink wash · Yours: studio glow")).not.toBeInTheDocument();
    expect(screen.getByText("From the file: ink wash")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Trigger words for Atelier Ink"), {
      target: { value: "studio glow, soft edge" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(api.updateModelAsset).toHaveBeenCalledWith("asset-ink", {
      typed_trigger_words: ["studio glow", "soft edge"],
    }));
  });
});

describe("base model of an installed LoRA", () => {
  const stamp = "2026-09-28T00:00:00Z";
  function lora(id: string, name: string, family: string | null) {
    return {
      id,
      source_id: null,
      name,
      kind: "lora" as const,
      family,
      size_bytes: 1024,
      manifest_json: {},
      active: true,
      use_case: "",
      auto_apply: false,
      default_model_strength: 1,
      default_clip_strength: 1,
      typed_trigger_words: [],
      verified_at: stamp,
      created_at: stamp,
      updated_at: stamp,
    };
  }

  async function editAutoRules(assets: ReturnType<typeof lora>[], name: string) {
    vi.mocked(api.modelAssets).mockResolvedValue(assets);
    vi.mocked(api.updateModelAsset).mockReset().mockResolvedValue(assets[0]);
    show();
    await screen.findByText(name);
    const index = assets.findIndex((asset) => asset.name === name);
    fireEvent.click(screen.getAllByRole("button", { name: "Edit Auto rules" })[index]);
    return screen.getByLabelText(`Base model for ${name}`);
  }

  it("records a base model that the file did not declare", async () => {
    const field = await editAutoRules([lora("asset-ink", "Atelier Ink", null)], "Atelier Ink");
    fireEvent.change(field, { target: { value: "  krea2 " } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(api.updateModelAsset).toHaveBeenCalledWith("asset-ink", {
      use_case: "",
      family: "krea2",
      auto_apply: false,
      default_model_strength: 1,
      default_clip_strength: 1,
    }));
  });

  it("clears the base model when the field is emptied", async () => {
    const field = await editAutoRules([lora("asset-ink", "Atelier Ink", "sdxl")], "Atelier Ink");
    expect(field).toHaveValue("sdxl");
    fireEvent.change(field, { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(api.updateModelAsset).toHaveBeenCalledWith(
      "asset-ink",
      expect.objectContaining({ family: "" }),
    ));
  });

  it("leaves the recorded base model alone when only another rule changes", async () => {
    await editAutoRules([lora("asset-ink", "Atelier Ink", " Krea-2")], "Atelier Ink");
    fireEvent.change(screen.getByLabelText("Auto use case for Atelier Ink"), {
      target: { value: "Ink wash portraits" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(api.updateModelAsset).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.updateModelAsset).mock.calls[0][1]).not.toHaveProperty("family");
  });

  it("offers the base models already recorded, one spelling of each", async () => {
    const field = await editAutoRules([
      lora("asset-ink", "Atelier Ink", null),
      lora("asset-glow", "Studio Glow", "sdxl"),
      lora("asset-wash", "Soft Wash", "krea2"),
      lora("asset-edge", "Hard Edge", "Krea-2"),
    ], "Atelier Ink");
    const list = document.getElementById(field.getAttribute("list") ?? "");
    const offered = [...(list?.querySelectorAll("option") ?? [])].map((option) => option.value);

    expect(offered).toEqual(["krea2", "sdxl"]);
  });

  it("refuses a base model with no letter or digit and says why", async () => {
    const field = await editAutoRules([lora("asset-ink", "Atelier Ink", null)], "Atelier Ink");
    fireEvent.change(field, { target: { value: "--" } });

    expect(screen.getByText("A base model needs at least one letter or digit.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });

  it("says a LoRA used automatically is never chosen without a base model", async () => {
    await editAutoRules([lora("asset-ink", "Atelier Ink", null)], "Atelier Ink");
    const note = "Without a base model, it is never chosen automatically.";
    expect(screen.queryByText(note)).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Use Atelier Ink automatically"));
    expect(screen.getByText(note)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Base model for Atelier Ink"), {
      target: { value: "sdxl" },
    });
    expect(screen.queryByText(note)).not.toBeInTheDocument();
  });
});
