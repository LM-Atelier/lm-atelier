/** Reading a generation record, and fetching it as the exact bytes the server wrote. */

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  generationRecordFileName,
  missingText,
  omissionText,
  readGenerationRecord,
  removedText,
} from "./generationRecord";
import { recordBytes } from "./generationRecordFixtures";

const SHA = "a".repeat(64);

describe("reading a record", () => {
  it("summarizes what the record says without changing it", () => {
    const bytes = recordBytes();
    const before = new Uint8Array(bytes).slice();

    const summary = readGenerationRecord(bytes);

    expect(summary).toEqual({
      digest: `sha256:${"d".repeat(64)}`,
      outputSha256: SHA,
      kind: "image",
      operation: "image_to_image",
      promptIncluded: false,
      promptOmittedReason: "chosen",
      seed: 42,
      seedBinding: "bound",
      settingCount: 2,
      inputCount: 1,
      inputBytes: 10,
      workflowVerified: true,
      modelFileCount: 1,
      loraCount: 0,
      removed: ["prompt", "settings.house_style"],
      missing: ["frozen_snapshot_absent", "prompt_omitted"],
    });
    expect(new Uint8Array(bytes)).toEqual(before);
    expect(generationRecordFileName(summary)).toBe(`generation-record-${SHA.slice(0, 12)}.json`);
  });

  it("refuses anything that is not a version 1 record", () => {
    expect(() => readGenerationRecord(recordBytes({ schema: "something-else" }))).toThrow();
    expect(() => readGenerationRecord(recordBytes({ version: 2 }))).toThrow();
    expect(() => readGenerationRecord(recordBytes({ prompt: "words" }))).toThrow();
  });

  it("words every left-out field and every shortfall, including ones it does not know", () => {
    expect(removedText("prompt")).toBe("The prompt");
    expect(removedText("settings.house_style")).toBe('The setting "house_style"');
    expect(removedText("settings.mask")).toBe("How the selection was applied");
    expect(missingText("finished_after_generation")).toMatch(/finished after the workflow ran/);
    expect(missingText("a_reason_from_a_later_version")).toBe("Something this record cannot vouch for.");
    expect(omissionText("removed_from_chat")).toMatch(/removed from the chat/);
    expect(omissionText(null)).toBe("Left out.");
  });
});

describe("fetching a record", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it("asks for an explicit prompt choice and returns the body byte for byte", async () => {
    const body = recordBytes();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
      .mockResolvedValueOnce(new Response(body.slice(0), { status: 200 }))
      .mockResolvedValueOnce(new Response(body.slice(0), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const { api } = await import("./api");
    const controller = new AbortController();

    const omitted = await api.generationRecord("run_1", `sha256:${SHA}`, false, controller.signal);
    await api.generationRecord("run_1", `sha256:${SHA}`, true);

    expect(new Uint8Array(omitted)).toEqual(new Uint8Array(body));
    expect(fetchMock.mock.calls[1][0]).toBe(`/api/runs/run_1/outputs/sha256%3A${SHA}/recipe?prompts=omit`);
    expect(fetchMock.mock.calls[1][1]?.signal).toBe(controller.signal);
    expect(fetchMock.mock.calls[1][1]?.method).toBeUndefined();
    expect(fetchMock.mock.calls[2][0]).toBe(`/api/runs/run_1/outputs/sha256%3A${SHA}/recipe?prompts=include`);
  });

  it("reports a refusal with its code", async () => {
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
      .mockResolvedValueOnce(new Response(
        JSON.stringify({ detail: "This generation has no such output.", code: "output-recipe-output-not-found" }),
        { status: 404 },
      )));
    const { api } = await import("./api");

    await expect(api.generationRecord("run_1", `sha256:${SHA}`, false)).rejects.toMatchObject({
      status: 404,
      code: "output-recipe-output-not-found",
    });
  });
});
