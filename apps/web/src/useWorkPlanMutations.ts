import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";

export function useWorkPlanMutations(activeChatId: string | null) {
  const client = useQueryClient();
  const refreshChatPlan = (chatId: string | null) => {
    void client.invalidateQueries({ queryKey: ["chat", chatId] });
    void client.invalidateQueries({ queryKey: ["work-plans", chatId] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
  };
  const cancelWorkPlan = useMutation({
    mutationFn: api.cancelWorkPlan,
    onSuccess: (plan) => refreshChatPlan(plan.chat_id),
  });
  const retryWorkPlan = useMutation({
    mutationFn: api.retryWorkPlan,
    onSuccess: (plan) => refreshChatPlan(plan.chat_id),
  });
  const cancelWorkStep = useMutation({
    mutationFn: api.cancelWorkStep,
    onSuccess: () => refreshChatPlan(activeChatId),
  });
  const retryWorkStep = useMutation({
    mutationFn: api.retryWorkStep,
    onSuccess: () => refreshChatPlan(activeChatId),
  });
  return { cancelWorkPlan, retryWorkPlan, cancelWorkStep, retryWorkStep };
}
