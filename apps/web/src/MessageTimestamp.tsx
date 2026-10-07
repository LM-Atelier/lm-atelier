import { clockOptions, useClockChoice } from "./clockPreference";

export function MessageTimestamp({ at }: { at: string }) {
  const clock = useClockChoice();
  const date = new Date(at);
  if (Number.isNaN(date.getTime())) return null;
  const sameDay = date.toDateString() === new Date().toDateString();
  const label = sameDay
    ? date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", ...clockOptions(clock) })
    : date.toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        ...clockOptions(clock),
      });
  return (
    <time className="message-timestamp" dateTime={at} title={date.toLocaleString([], clockOptions(clock))}>
      {label}
    </time>
  );
}
