import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "./api";

// Deleting a turn removes messages, runs, jobs, a possible work plan, and
// artifact references in one server-side cascade, so every affected query is
// invalidated together here rather than at each call site.
export function useExchangeDeletion() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (messageId: string) => api.deleteExchange(messageId),
    onSuccess: (result) => {
      void client.invalidateQueries({ queryKey: ["chat", result.chat_id] });
      void client.invalidateQueries({ queryKey: ["work-plans", result.chat_id] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["artifacts"] });
    },
  });
}
