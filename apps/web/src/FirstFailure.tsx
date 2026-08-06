import { ErrorCallout } from "./ErrorCallout";
import type { FailingMutation } from "./GlobalNotices";

/** The first failure among several requests, reported once.
 *
 * The alternative is the hand-written `||` chain, which this codebase has
 * already been bitten by twice: one omitted a mutation entirely, and the
 * longest of them ran to eight names on a single line, where a ninth that
 * belonged was impossible to notice missing. A list is checkable; a chain is
 * only readable.
 *
 * Queries are welcome here too. An unreadable list and a refused action are
 * the same thing to the person looking at the screen.
 */
export function FirstFailure({ of }: { of: FailingMutation[] }) {
  const failure = of.find((source) => source.error)?.error;
  return failure ? <ErrorCallout message={failure.message} /> : null;
}
