import { describe, expect, it } from "vitest";
import { sizeDisagreement, sizeDisagreementMessage } from "./outputSizeAgreement";
import type { MessagePart } from "./types";

/** The judgement belongs to the PART, not to the artifact: identical bytes from
 * two runs are one shared artifact, and a judgement about what was asked for is
 * not a property of the bytes. */
function artifact(agreement: unknown): MessagePart {
  return {
    id: "part-1",
    position: 0,
    type: "image",
    text: null,
    artifact_id: "sha256:image",
    metadata_json: agreement === undefined ? {} : { output_size_agreement: agreement },
  };
}

describe("the recorded size judgement", () => {
  it("reports the two sizes when they genuinely differ", () => {
    expect(
      sizeDisagreement(
        artifact({
          v: 1,
          state: "disagreed",
          requested_width: 1024,
          requested_height: 768,
          raster_width: 2048,
          raster_height: 1536,
        }),
      ),
    ).toEqual({
      requestedWidth: 1024,
      requestedHeight: 768,
      actualWidth: 2048,
      actualHeight: 1536,
    });
  });

  it("says nothing when the picture is the size that was asked for", () => {
    expect(
      sizeDisagreement(
        artifact({
          v: 1,
          state: "agreed",
          requested_width: 1024,
          requested_height: 768,
          raster_width: 1024,
          raster_height: 768,
        }),
      ),
    ).toBeNull();
  });

  // Each of these is a state where the tool never formed an opinion. They
  // outnumber the one state that should warn, which is the whole reason this
  // reader exists rather than a `!== "agreed"` at the call site.
  it.each([
    ["throwaway", "the file was a preview node's, never the picture in question"],
    ["origin_unknown", "the engine named nothing, so we cannot say which node wrote it"],
    ["no_size_requested", "nobody asked for a size"],
    ["binding_unconfirmed", "we could not confirm width and height decide the output"],
    ["unmeasured", "the picture could not be measured"],
  ])("stays silent when the reason is %s", (reason) => {
    expect(sizeDisagreement(artifact({ v: 1, state: "not_assessed", reason }))).toBeNull();
  });

  // The state decides, and this is the test that proves it does. Every other
  // non-assessment case above also lacks dimensions, so the dimension guard
  // alone would satisfy them - and a reader that warned on any state except
  // "agreed" would pass all of them while being wrong. Today's producer never
  // writes numbers beside a non-assessment, but a later one might well record
  // what was REQUESTED even when it could not measure, and that must not start
  // warning people.
  it("stays silent on a non-assessment that also carries numbers", () => {
    expect(
      sizeDisagreement(
        artifact({
          v: 1,
          state: "not_assessed",
          reason: "binding_unconfirmed",
          requested_width: 1024,
          requested_height: 768,
          raster_width: 2048,
          raster_height: 1536,
        }),
      ),
    ).toBeNull();
  });

  it.each([
    ["no record at all", undefined],
    ["a record that is not an object", "disagreed"],
    ["null", null],
  ])("stays silent given %s", (_label, value) => {
    expect(sizeDisagreement(artifact(value))).toBeNull();
  });

  it("stays silent on an artifact that is absent", () => {
    expect(sizeDisagreement(null)).toBeNull();
    expect(sizeDisagreement(undefined)).toBeNull();
  });

  // A half-filled warning beside a picture is worse than none: it would show a
  // sentence with a hole in it, or invite the caller to invent the missing half.
  it.each([
    ["a missing number", { raster_height: undefined }],
    ["a non-integer", { raster_height: 768.5 }],
    ["zero", { raster_height: 0 }],
    ["a negative", { raster_height: -768 }],
    ["a string that looks like a number", { raster_height: "768" }],
  ])("refuses to warn when a dimension is %s", (_label, override) => {
    expect(
      sizeDisagreement(
        artifact({
          v: 1,
          state: "disagreed",
          requested_width: 1024,
          requested_height: 768,
          raster_width: 2048,
          ...override,
        }),
      ),
    ).toBeNull();
  });
});

describe("the sentence", () => {
  it("says what arrived and what was asked for, and nothing else", () => {
    expect(
      sizeDisagreementMessage({
        requestedWidth: 1024,
        requestedHeight: 768,
        actualWidth: 2048,
        actualHeight: 1536,
      }),
    ).toBe("This came out 2048 × 1536, not the 1024 × 768 you asked for.");
  });

  it("offers no explanation, because there is none to offer", () => {
    const message = sizeDisagreementMessage({
      requestedWidth: 512,
      requestedHeight: 512,
      actualWidth: 768,
      actualHeight: 512,
    });
    for (const guess of ["because", "workflow", "try", "sorry", "failed", "error"]) {
      expect(message.toLowerCase()).not.toContain(guess);
    }
  });
});
