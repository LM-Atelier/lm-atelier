import { adjustmentAnswerer, type AdjustWorkAnswer, type AdjustWorkMessage } from "./studioAdjustWork";

/** The worker that works out the light and color preview beside the page.
 *
 * All it does is hand each message to the answerer in studioAdjustWork.ts and
 * send any answer back, moving the adjusted pixels rather than copying them.
 */

const scope = globalThis as unknown as {
  onmessage: ((event: MessageEvent<AdjustWorkMessage>) => void) | null;
  postMessage: (message: AdjustWorkAnswer, transfer: Transferable[]) => void;
};
const answer = adjustmentAnswerer();

scope.onmessage = (event) => {
  const reply = answer(event.data);
  if (reply) scope.postMessage(reply, [reply.pixels.buffer]);
};
