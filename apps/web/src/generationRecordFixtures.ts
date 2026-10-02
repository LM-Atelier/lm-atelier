/** A neutral generation record for tests, as the server would write it. */

const SHA = "a".repeat(64);

export function recordText(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    digest: `sha256:${"d".repeat(64)}`,
    exported_by: { application: "LM Atelier", version: "0.2.0" },
    inputs: [{ media_type: "image/png", role: "source", sha256: "b".repeat(64), size_bytes: 10 }],
    loras: [],
    model: { content_rating: null, files: { "model.safetensors": "c".repeat(64) }, provider: null, remote_id: null, revision: null },
    not_recorded: ["executed_graph_sha256", "runtime_version"],
    operation: "image_to_image",
    output: { collection: "images", count: 1, engine: "comfyui", index: 0, kind: "image", media_type: "image/png", node_id: "9", raster: null, sha256: SHA, size_bytes: 20 },
    prompt: { included: false, negative: null, omitted_reason: "chosen", positive: null },
    removed: ["prompt", "settings.house_style"],
    reproducibility: { missing: ["frozen_snapshot_absent", "prompt_omitted"], status: "incomplete" },
    schema: "lm-atelier-output-recipe-v1",
    seed: { binding: "bound", value: 42 },
    settings: { bound: { steps: 20 }, unbound: { sampler: "euler" } },
    version: 1,
    workflow: { artifact_sha256: "e".repeat(64), binding_sha256: null, contract_version: 1, dependency_contract_sha256: null, engine: "comfyui", graph_source: "live", operation: "image_to_image", verified: true },
    ...overrides,
  });
}

export function recordBytes(overrides: Record<string, unknown> = {}): ArrayBuffer {
  const encoded = new TextEncoder().encode(recordText(overrides));
  return encoded.buffer.slice(encoded.byteOffset, encoded.byteOffset + encoded.byteLength);
}
