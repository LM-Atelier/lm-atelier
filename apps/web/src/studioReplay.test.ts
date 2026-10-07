/** Try another: the same edit again, read from its run, with a new random seed. */

import { describe, expect, it } from "vitest";
import { studioReplay } from "./studioReplay";
import type { Run } from "./types";

function run(provenance: Record<string, unknown>, workflowRevisionId: string | null = null): Run {
  return { id: "run-7", workflow_revision_id: workflowRevisionId, provenance_json: provenance } as unknown as Run;
}

describe("the same edit again", () => {
  it("keeps what the run resolved and the selection, and asks for a new seed", () => {
    const mask = { artifact_id: "mask-1", feather_px: 4, invert: false };
    const replay = studioReplay(
      run({
        resolved_settings: { steps: 20, denoise: 0.6, seed: 1234, noise_seed: 99, batch_size: 1, mask },
        workflow: { revision_id: "rev-edit" },
      }),
      "make the sky warmer",
      ["art-1"],
    );

    expect(replay).toEqual({
      words: "make the sky warmer",
      inputs: ["art-1"],
      settings: { steps: 20, denoise: 0.6, mask, seed: -1 },
      workflowRevisionId: "rev-edit",
    });
  });

  it("sends every picture the edit was given, the one it changed first", () => {
    const replay = studioReplay(run({ resolved_settings: {} }), "relight it", ["art-1", "light-map"]);

    expect(replay?.inputs).toEqual(["art-1", "light-map"]);
  });

  it("takes the run's own workflow when its record names none", () => {
    expect(studioReplay(run({ resolved_settings: {} }, "rev-run"), "w", ["art-1"])?.workflowRevisionId).toBe("rev-run");
    expect(studioReplay(run({ resolved_settings: {}, workflow: { revision_id: "" } }), "w", ["art-1"])).not.toHaveProperty(
      "workflowRevisionId",
    );
  });

  it("makes nothing again from a run that recorded no settings, or with no picture to change", () => {
    expect(studioReplay(run({}), "w", ["art-1"])).toBeNull();
    expect(studioReplay(run({ resolved_settings: ["not", "settings"] }), "w", ["art-1"])).toBeNull();
    expect(studioReplay(run({ resolved_settings: {} }), "w", [])).toBeNull();
  });

  it("says an enlargement again as one, sending its factor only where the run resolved one", () => {
    const fixed = studioReplay(run({ resolved_settings: { steps: 1 }, upscale: true }, "rev-enlarge"), "w", ["art-1"]);
    const plain = studioReplay(run({ resolved_settings: { steps: 1 } }, "rev-edit"), "w", ["art-1"]);

    expect(fixed).toEqual({ words: "w", inputs: ["art-1"], settings: { steps: 1, seed: -1 }, workflowRevisionId: "rev-enlarge", upscale: true });
    expect(plain).not.toHaveProperty("upscale");
  });
});
