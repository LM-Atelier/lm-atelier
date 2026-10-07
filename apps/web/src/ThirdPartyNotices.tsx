import { useQuery } from "@tanstack/react-query";
import { Folder } from "lucide-react";
import { useState } from "react";
import { api } from "./api";
import { CopyTextButton } from "./CopyTextButton";
import { ErrorCallout } from "./ErrorCallout";
import { MarkdownText } from "./MarkdownText";

/** The third-party software a release includes, read only when somebody opens it.
 *
 * Releases carry the inventory and the license texts next to the application.
 * A copy run from source has neither, and says where to find them instead of
 * showing an empty list.
 */
export function ThirdPartyNotices() {
  const [open, setOpen] = useState(false);
  const notices = useQuery({
    queryKey: ["about", "third-party-notices"],
    queryFn: api.thirdPartyNotices,
    enabled: open,
  });
  return (
    <details className="about-notices" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Third-party notices</summary>
      {open && notices.isPending && <p role="status">Reading the notices…</p>}
      {notices.data?.text && (
        <div className="about-notices-inventory">
          <MarkdownText text={notices.data.text} />
        </div>
      )}
      {notices.data?.license_folder && (
        <div className="about-paths">
          <div>
            <Folder size={17} />
            <span><small>License texts</small><code>{notices.data.license_folder}</code></span>
            <CopyTextButton
              text={notices.data.license_folder}
              label="Copy license texts folder"
              buttonText="Copy folder"
              className="secondary compact-button"
            />
          </div>
        </div>
      )}
      {notices.data && !notices.data.text && (
        <p className="muted">
          Installed releases of LM Atelier include the third-party notices and their license texts, next to the application.
        </p>
      )}
      {notices.error && <ErrorCallout message="Third-party notices are unavailable right now." />}
    </details>
  );
}
