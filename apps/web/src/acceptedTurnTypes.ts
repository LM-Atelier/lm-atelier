import type { Message, Run } from "./types";

export interface TurnAccepted {
  run: Run;
  user_message: Message;
  assistant_message: Message;
  assistant_messages?: Message[];
  /** True when the idempotency key matched an earlier submission in this chat and
   * this is the run it created; false when this request created the run. */
  replayed: boolean;
}
