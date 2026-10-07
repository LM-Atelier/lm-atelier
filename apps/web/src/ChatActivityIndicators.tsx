import { Circle, CircleAlert, LoaderCircle } from "lucide-react";
import type { ChatActivity } from "./types";
import { useChatActivitySeen } from "./useChatActivitySeen";

export function ChatActivityIndicators({ chatId, activity }: { chatId: string; activity: ChatActivity }) {
  const seen = useChatActivitySeen();
  const active = activity.active_work_count;
  const failures = activity.unresolved_failed_count;
  const unread = activity.last_output && !seen.hasSeen(chatId, activity.last_output);
  const count = (value: number) => value > 99 ? "99+" : String(value);
  return <span className="chat-activity-indicators">
    {active > 0 && <span role="img" aria-label={`${active} active ${active === 1 ? "task" : "tasks"}`} title={`${active} active tasks`}><LoaderCircle className="spin" size={13} aria-hidden="true" />{count(active)}</span>}
    {failures > 0 && <span role="img" aria-label={`${failures} unresolved ${failures === 1 ? "failure" : "failures"}`} title={`${failures} unresolved failures`}><CircleAlert size={13} aria-hidden="true" />{count(failures)}</span>}
    {unread && <span role="img" aria-label="Unread output" title="Unread output"><Circle size={8} fill="currentColor" aria-hidden="true" /></span>}
  </span>;
}
