import type { Message, Run } from "./types";

export interface TurnAccepted {
  run: Run;
  user_message: Message;
  assistant_message: Message;
  assistant_messages?: Message[];
}
