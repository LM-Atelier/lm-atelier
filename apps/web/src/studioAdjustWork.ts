import { adjustPixels } from "./studioAdjustments";
import type { StudioColorAdjustments } from "./types";

/** Light and color worked out beside the page: the messages, the worker's side and the page's side.
 *
 * The preview's arithmetic takes a third of a second or more on a camera-sized
 * picture, and on the page that time is taken from everything else the page
 * does, so a slider dragged across it stutters. A worker does the same
 * arithmetic beside the page. The picture's pixels go to it once, when the
 * picture changes; after that only the sliders do, and each answer comes back
 * as the adjusted pixels, moved rather than copied.
 */

/** What the page sends: the picture, once, then each setting of the sliders for it. */
export type AdjustWorkMessage =
  | { kind: "picture"; picture: number; pixels: Uint8ClampedArray; width: number }
  | { kind: "adjust"; picture: number; request: number; adjustments: StudioColorAdjustments };

/** What comes back: one setting's adjusted pixels. */
export type AdjustWorkAnswer = { picture: number; request: number; pixels: Uint8ClampedArray };

/** The worker's side: hold the picture last sent, and answer each setting for it.
 *
 * The answer is the page's own arithmetic, so it is the picture Apply makes. A
 * setting for any other picture is not answered, so no answer can belong to a
 * picture the page has moved on from.
 */
export function adjustmentAnswerer(): (message: AdjustWorkMessage) => AdjustWorkAnswer | null {
  let held: { picture: number; pixels: Uint8ClampedArray; width: number } | null = null;
  return (message) => {
    if (message.kind === "picture") {
      held = { picture: message.picture, pixels: message.pixels, width: message.width };
      return null;
    }
    if (!held || held.picture !== message.picture) return null;
    return {
      picture: message.picture,
      request: message.request,
      pixels: adjustPixels(held.pixels, held.width, message.adjustments),
    };
  };
}

/** The page's side: one setting in flight at a time, and the newest waiting behind it.
 *
 * A slider dragged faster than the worker answers would otherwise queue a
 * setting for every step, each a third of a second long. Instead the newest
 * setting waits, and goes when the one in flight is answered, so the preview
 * follows the slider as fast as it can be made and ends where the slider
 * stopped. An answer is shown only while it is the newest one asked for the
 * picture held.
 */
export class AdjustmentWork {
  private picture = 0;
  private request = 0;
  private inFlight = false;
  private waiting: StudioColorAdjustments | null = null;

  constructor(
    private readonly send: (message: AdjustWorkMessage) => void,
    private readonly show: (picture: number, pixels: Uint8ClampedArray) => void,
  ) {}

  /** A new picture, sent once; every setting after it is for this one. Returns its number. */
  hold(pixels: Uint8ClampedArray, width: number): number {
    this.picture += 1;
    this.forget();
    this.send({ kind: "picture", picture: this.picture, pixels, width });
    return this.picture;
  }

  /** The sliders' newest setting for the picture held. */
  adjust(adjustments: StudioColorAdjustments): void {
    if (this.inFlight) {
      this.waiting = adjustments;
      return;
    }
    this.inFlight = true;
    this.request += 1;
    this.send({ kind: "adjust", picture: this.picture, request: this.request, adjustments });
  }

  /** Nothing asked so far is to be shown. */
  forget(): void {
    this.request += 1;
    this.inFlight = false;
    this.waiting = null;
  }

  /** An answer from the worker: shown if it is still the newest, then the waiting setting goes. */
  answered(answer: AdjustWorkAnswer): void {
    if (answer.picture !== this.picture || answer.request !== this.request) return;
    this.inFlight = false;
    this.show(answer.picture, answer.pixels);
    const next = this.waiting;
    this.waiting = null;
    if (next) this.adjust(next);
  }
}
