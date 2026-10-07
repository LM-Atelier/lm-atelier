import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { LoraStackControl } from "./LoraStackControl";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, modelAssets: vi.fn() } };
});

afterEach(cleanup);

it("lists a LoRA's typed trigger words after the ones its file declares", async () => {
  const stamp = "2026-09-23T00:00:00Z";
  vi.mocked(api.modelAssets).mockResolvedValue([{
    id: "asset-ink",
    source_id: null,
    name: "Atelier Ink",
    kind: "lora",
    family: "sdxl",
    size_bytes: 1024,
    manifest_json: { metadata: { trigger_words: ["ink wash"] } },
    active: true,
    use_case: "",
    auto_apply: false,
    default_model_strength: 1,
    default_clip_strength: 1,
    typed_trigger_words: ["studio glow", "Ink Wash"],
    verified_at: stamp,
    created_at: stamp,
    updated_at: stamp,
  }]);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <LoraStackControl
        value={[{ asset_id: "asset-ink", model_strength: 1, clip_strength: 1, enabled: true }]}
        onChange={vi.fn()}
      />
    </QueryClientProvider>,
  );

  // A typed word the file already declares, in another casing, is shown once.
  expect(await screen.findByText("sdxl · ink wash · studio glow")).toBeInTheDocument();
});
