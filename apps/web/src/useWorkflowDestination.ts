import { useState } from "react";

export type WorkflowDestination = "library" | "discover";

/** Which half of the Workflows page is showing, and whether Discover has opened yet.
 *
 * The page keeps the library HIDDEN rather than unmounted while Discover is
 * open. Its children hold work that has not been saved anywhere - the family
 * list's search and filters among them - and unmounting would discard that as
 * a side effect of merely looking elsewhere.
 *
 * Discover mounts on its first visit and stays mounted afterwards. Mounting it
 * lazily means opening your own library never sends a request to a remote,
 * rate-limited source nobody asked about; keeping it afterwards means a search
 * is still there when you come back to it.
 */
export function useWorkflowDestination() {
  const [destination, setDestination] = useState<WorkflowDestination>("library");
  const [discoverVisited, setDiscoverVisited] = useState(false);
  const show = (next: WorkflowDestination) => {
    if (next === "discover") setDiscoverVisited(true);
    setDestination(next);
  };
  return { destination, discoverVisited, show };
}
