import type { ReactNode } from "react";
import type { ChatActivityReference } from "./types";

export function VisibleChatActivity({ activity, children }: {
  activity: ChatActivityReference | null | undefined;
  children: ReactNode;
}) {
  return activity ? <div data-chat-activity={JSON.stringify(activity)}>{children}</div> : children;
}
