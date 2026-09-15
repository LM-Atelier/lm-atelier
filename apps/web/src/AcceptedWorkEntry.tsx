import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ListChecks } from "lucide-react";
import { QueueActivityDialog } from "./QueueActivityDialog";

/** Opens accepted work from a fixed place in the sidebar.
 *
 * A floating corner button covered the composer wherever the chat column
 * reached the window edge, and the room reserved beneath the workspace to
 * avoid that took space from every view. The dialog belongs to this entry,
 * so jobs starting or finishing never close it, and closing it returns focus
 * here.
 */
export function AcceptedWorkEntry() {
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  const close = () => {
    setOpen(false);
    button.current?.focus();
  };
  return (
    <>
      <button ref={button} onClick={() => setOpen(true)}><ListChecks />Accepted work</button>
      {open && createPortal(<QueueActivityDialog onClose={close} />, document.body, "accepted-work")}
    </>
  );
}
