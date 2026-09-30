/** Light and color worked out beside the page: what the worker answers and what the page asks. */

import { describe, expect, it } from "vitest";
import { adjustPixels, NEUTRAL_ADJUSTMENTS } from "./studioAdjustments";
import { AdjustmentWork, adjustmentAnswerer, type AdjustWorkMessage } from "./studioAdjustWork";

const PIXELS = new Uint8ClampedArray([200, 100, 50, 255, 10, 240, 128, 255, 128, 128, 128, 255, 37, 91, 203, 255]);
const BRIGHTER = { ...NEUTRAL_ADJUSTMENTS, brightness: 40 };
const GREY = { ...NEUTRAL_ADJUSTMENTS, saturation: -100 };
const SHARPER = { ...NEUTRAL_ADJUSTMENTS, sharpness: 60 };

describe("the worker's side", () => {
  it("answers a setting for the picture it holds with the page's own arithmetic", () => {
    const answer = adjustmentAnswerer();
    expect(answer({ kind: "picture", picture: 1, pixels: PIXELS, width: 2 })).toBeNull();

    const reply = answer({ kind: "adjust", picture: 1, request: 3, adjustments: BRIGHTER });

    expect(reply?.picture).toBe(1);
    expect(reply?.request).toBe(3);
    expect(Array.from(reply?.pixels ?? [])).toEqual(Array.from(adjustPixels(PIXELS, 2, BRIGHTER)));
  });

  it("answers nothing for a picture it does not hold", () => {
    const answer = adjustmentAnswerer();
    expect(answer({ kind: "adjust", picture: 1, request: 1, adjustments: BRIGHTER })).toBeNull();
    answer({ kind: "picture", picture: 2, pixels: PIXELS, width: 2 });
    expect(answer({ kind: "adjust", picture: 1, request: 2, adjustments: BRIGHTER })).toBeNull();
  });
});

describe("the page's side", () => {
  function recorded() {
    const sent: AdjustWorkMessage[] = [];
    const shown: Array<[number, Uint8ClampedArray]> = [];
    const work = new AdjustmentWork(
      (message) => sent.push(message),
      (picture, pixels) => shown.push([picture, pixels]),
    );
    return { sent, shown, work };
  }

  it("keeps one setting in flight, and sends only the newest one waiting when it is answered", () => {
    const { sent, shown, work } = recorded();
    const picture = work.hold(PIXELS, 2);
    work.adjust(BRIGHTER);
    work.adjust(GREY);
    work.adjust(SHARPER);

    expect(sent.map((message) => message.kind)).toEqual(["picture", "adjust"]);
    const first = sent[1];
    if (first.kind !== "adjust") throw new Error("a setting is sent after the picture");
    expect(first.adjustments).toEqual(BRIGHTER);

    const answer = new Uint8ClampedArray(16);
    work.answered({ picture, request: first.request, pixels: answer });

    // Shown while the slider moves on, and the setting sent next is where it stopped.
    expect(shown).toEqual([[picture, answer]]);
    expect(sent).toHaveLength(3);
    const next = sent[2];
    if (next.kind !== "adjust") throw new Error("the waiting setting goes next");
    expect(next.adjustments).toEqual(SHARPER);
  });

  it("shows no answer for a picture it has moved on from, or one asked before it forgot", () => {
    const { sent, shown, work } = recorded();
    const first = work.hold(PIXELS, 2);
    work.adjust(BRIGHTER);
    const early = sent[1];
    if (early.kind !== "adjust") throw new Error("a setting is sent after the picture");

    work.hold(PIXELS, 2);
    work.answered({ picture: first, request: early.request, pixels: new Uint8ClampedArray(16) });
    expect(shown).toEqual([]);

    const second = sent.length;
    work.adjust(GREY);
    const late = sent[second];
    if (late.kind !== "adjust") throw new Error("a setting is sent for the new picture");
    work.forget();
    work.answered({ picture: late.picture, request: late.request, pixels: new Uint8ClampedArray(16) });
    expect(shown).toEqual([]);
  });
});
