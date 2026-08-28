import { describe, expect, it } from "vitest";
import { videoLengthSummary } from "./videoLength";

describe("videoLengthSummary", () => {
  it("shows both requested and delivered duration when alignment changes it", () => {
    expect(videoLengthSummary({
      video_length: { requested_seconds: 3, delivered_seconds: 49 / 16 },
    })).toBe("Requested 3s · delivered 3.0625s");
  });

  it("shows only delivered duration when the request is exact", () => {
    expect(videoLengthSummary({
      video_length: { requested_seconds: 2.5, delivered_seconds: 2.5 },
    })).toBe("Delivered 2.5s");
  });

  it("does not invent a summary from malformed provenance", () => {
    expect(videoLengthSummary(undefined)).toBeNull();
    expect(videoLengthSummary({ video_length: { requested_seconds: "many" } })).toBeNull();
  });
});
