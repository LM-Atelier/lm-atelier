import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import {
  REPLAYABLE_OPERATIONS,
  replayFailureText,
  replayReasonText,
  replayRefusalText,
  type ReplayPlan,
} from "./generationRecord";

/** Whether a checked record can be generated again exactly here, and the way to do it. */
export function GenerationRecordReplay({
  plan,
  content,
  onStarted,
}: {
  plan: ReplayPlan;
  content: ArrayBuffer;
  onStarted: (chatId: string) => void;
}) {
  const client = useQueryClient();
  const replay = useMutation({
    mutationFn: async () => {
      const chat = await api.createChat(null);
      try {
        await api.replayGenerationRecord(chat.id, content);
      } catch (error) {
        // The new chat holds nothing yet, so it goes rather than stays behind empty.
        await api.deleteChat(chat.id).catch(() => undefined);
        throw error;
      }
      return chat.id;
    },
    onSuccess: (chatId) => {
      void client.invalidateQueries({ queryKey: ["chats"] });
      onStarted(chatId);
    },
  });
  const replayable = plan.ready && REPLAYABLE_OPERATIONS.has(plan.operation);

  return (
    <section className="generation-record-body" aria-labelledby={`replay-${plan.digest}`}>
      <h3 id={`replay-${plan.digest}`}>Generate it again</h3>
      {replayable && (
        <p>Everything it names is here exactly, so it can be generated again in a new chat.</p>
      )}
      {plan.ready && !replayable && (
        <p>Everything it names is here, but this kind of generation cannot be started again yet.</p>
      )}
      {!plan.ready && (
        <>
          <p>It cannot be generated again exactly here:</p>
          <ul>
            {plan.refusals.map((refusal, index) => (
              <li key={`${refusal.code}:${refusal.sha256 ?? index}`}>
                {replayRefusalText(refusal.code)}
                {refusal.reasons.length > 0 && (
                  <ul>
                    {refusal.reasons.map((reason) => <li key={reason}>{replayReasonText(reason)}</li>)}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
      {replay.isPending && <p role="status">Starting it in a new chat…</p>}
      {replay.isError && <p role="alert">{replayFailureText(replay.error)}</p>}
      {replayable && (
        <div>
          <button
            type="button"
            className="primary"
            aria-disabled={replay.isPending}
            onClick={() => {
              if (!replay.isPending) replay.mutate();
            }}
          >
            Generate again
          </button>
        </div>
      )}
    </section>
  );
}
