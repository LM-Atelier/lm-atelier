import { useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";

export function PendingResponseStatus({ label, startedAt }: { label: string; startedAt: string }) {
  const [seconds, setSeconds] = useState(
    () => Math.max(0, Math.floor((Date.now() - Date.parse(startedAt)) / 1_000)),
  );
  useEffect(() => {
    const timer = window.setInterval(
      () => setSeconds(Math.max(0, Math.floor((Date.now() - Date.parse(startedAt)) / 1_000))),
      1_000,
    );
    return () => window.clearInterval(timer);
  }, [startedAt]);
  return (
    <div className="submission-progress pending-response" role="status">
      <LoaderCircle size={17} />
      <span>{label}<small aria-hidden="true"> · {seconds}s</small></span>
    </div>
  );
}
