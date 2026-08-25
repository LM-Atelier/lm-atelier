import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Quote } from "lucide-react";
import { api } from "./api";
import {
  admitCreatedPromptBatch,
  admitQueuedPromptBatch,
  promptCountMaximum,
  randomPromptSelectionSeed,
  type PromptDirectQueueAuthority,
} from "./promptDirectQueue";
import { PromptTemplatesDialog } from "./PromptTemplatesDialog";
import type {
  PromptBatch,
  PromptBatchCreateInput,
  PromptBatchQueueInput,
  PromptTemplateDetail,
} from "./types";

export type PromptDirectQueueStatus =
  | "creating"
  | "queueing"
  | "queued"
  | "error";

export interface PromptDirectQueueRequest {
  template: PromptTemplateDetail;
  itemCount: number;
  inputs: Record<string, string | string[]>;
}

export interface PromptDirectQueueAttempt {
  template: PromptTemplateDetail;
  createPayload: PromptBatchCreateInput;
  queueIdempotencyKey: string;
  batch: PromptBatch | null;
  status: PromptDirectQueueStatus;
  errorStage: "create" | "queue" | "admission" | null;
}

export function ComposerPromptTemplatesAction({
  chatId,
  currentPrompt,
  maximum,
}: {
  chatId: string;
  currentPrompt: string;
  maximum: number;
}) {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [attempt, setAttempt] = useState<PromptDirectQueueAttempt | null>(null);
  const attemptRef = useRef<PromptDirectQueueAttempt | null>(null);
  const inFlight = useRef(false);
  const countMaximum = promptCountMaximum(maximum);

  const publish = (next: PromptDirectQueueAttempt | null) => {
    attemptRef.current = next;
    setAttempt(next);
  };

  const complete = (queued: PromptBatch, current: PromptDirectQueueAttempt) => {
    publish({ ...current, batch: queued, status: "queued", errorStage: null });
    client.setQueryData(["prompt-batch", queued.id], queued);
    void client.invalidateQueries({ queryKey: ["chat", chatId] });
    void client.invalidateQueries({ queryKey: ["work-plans", chatId] });
  };

  const run = async (frozen: PromptDirectQueueAttempt) => {
    if (inFlight.current) return;
    inFlight.current = true;
    let current = frozen;
    let stage: "create" | "queue" = frozen.batch ? "queue" : "create";
    try {
      const authority: PromptDirectQueueAuthority = {
        chatId,
        templateId: frozen.template.id,
        revisionId: frozen.template.current_revision.id,
        schemaVersion: frozen.template.current_revision.schema_version,
        contractSha256: frozen.template.current_revision.contract_sha256,
        itemCount: frozen.createPayload.item_count,
        selectionSeed: frozen.createPayload.selection_seed,
        queueIdempotencyKey: frozen.queueIdempotencyKey,
      };
      let draft = frozen.batch;
      if (!draft) {
        current = { ...current, status: "creating", errorStage: null };
        publish(current);
        const response = await api.createPromptBatch(chatId, frozen.createPayload);
        draft = await admitCreatedPromptBatch(response, authority);
        current = { ...current, batch: draft };
        if (draft.state === "queued") {
          complete(draft, current);
          return;
        }
      }

      stage = "queue";
      current = { ...current, batch: draft, status: "queueing", errorStage: null };
      publish(current);
      const queuePayload: PromptBatchQueueInput = {
        idempotency_key: frozen.queueIdempotencyKey,
        expected_plan_version: draft.plan_version,
        expected_plan_sha256: draft.plan_sha256,
      };
      const response = await api.queuePromptBatch(draft.id, queuePayload);
      const queued = await admitQueuedPromptBatch(response, authority, draft);
      complete(queued, current);
    } catch (error) {
      const errorStage = error instanceof Error
        && error.name === "PromptDirectQueueAdmissionError"
        ? "admission"
        : stage;
      publish({ ...current, status: "error", errorStage });
    } finally {
      inFlight.current = false;
    }
  };

  const begin = (request: PromptDirectQueueRequest) => {
    if (
      inFlight.current
      || attemptRef.current
      || !Number.isInteger(request.itemCount)
      || request.itemCount < 1
      || request.itemCount > countMaximum
    ) return;
    const frozen: PromptDirectQueueAttempt = {
      template: structuredClone(request.template),
      createPayload: {
        idempotency_key: crypto.randomUUID(),
        template_revision_id: request.template.current_revision.id,
        contract_sha256: request.template.current_revision.contract_sha256,
        item_count: request.itemCount,
        selection_seed: randomPromptSelectionSeed(),
        inputs: structuredClone(request.inputs),
      },
      queueIdempotencyKey: crypto.randomUUID(),
      batch: null,
      status: "creating",
      errorStage: null,
    };
    publish(frozen);
    void run(frozen);
  };

  const retry = () => {
    const current = attemptRef.current;
    if (!current || current.status !== "error" || inFlight.current) return;
    void run(current);
  };

  const discard = () => {
    if (inFlight.current) return;
    publish(null);
  };

  const close = () => {
    setOpen(false);
    if (attemptRef.current?.status === "queued") publish(null);
  };

  return (
    <>
      <button
        type="button"
        className="icon-button"
        onClick={() => setOpen(true)}
        aria-label="Open prompt templates"
        title="Prompt templates"
      >
        <Quote size={18} />
      </button>
      {open && (
        <PromptTemplatesDialog
          currentPrompt={currentPrompt}
          maximum={countMaximum}
          attempt={attempt}
          onClose={close}
          onCreate={begin}
          onRetry={retry}
          onDiscard={discard}
        />
      )}
    </>
  );
}
