import { useState } from "react";

/** The installer's hand-off (--first-run-setup): setup before the workspace.
 *
 * The flag arrives as a query parameter on the installer's launch URL, and
 * exiting - finishing setup or explicitly skipping - clears it exactly once
 * so a reload lands in the normal application.
 */
export function useFirstRunSetup(): [boolean, () => void] {
  const [active, setActive] = useState(() => (
    new URLSearchParams(window.location.search).get("firstRunSetup") === "1"
  ));
  return [active, () => {
    window.history.replaceState(null, "", window.location.pathname);
    setActive(false);
  }];
}
