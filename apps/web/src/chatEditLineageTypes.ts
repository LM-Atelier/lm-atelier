export interface ChatEditLineageStep {
  message_id: string;
  artifact_id: string;
  instruction: string;
}

export interface ChatEditLineagePage {
  chat_id: string;
  result_message_id: string;
  steps: ChatEditLineageStep[];
  next_before: string | null;
}
