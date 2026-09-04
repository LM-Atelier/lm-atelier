import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpen, Plus, Search } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api, ApiError } from "./api";
import type {
  PromptDirectQueueAttempt,
  PromptDirectQueueRequest,
} from "./ComposerPromptTemplatesAction";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { PromptTemplateImageSetupPicker } from "./PromptTemplateImageSetupPicker";
import { promptTemplateImageSetupIsComplete } from "./promptTemplateImageSetup";
import type {
  PromptTemplateContract,
  PromptTemplateDetail,
  PromptTemplateResourcePolicy,
  PromptTemplateSlot,
} from "./types";

const TEMPLATE_LIMIT = 100;
const MAX_INPUT_CHARACTERS = 2_000;

type InputSlot = Extract<PromptTemplateSlot, { mode: "input" }>;

function quickTemplateContract(
  body: string,
  resourcePolicy: Extract<PromptTemplateResourcePolicy, { mode: "inherited" | "fixed" }>,
): PromptTemplateContract {
  return {
    schema_version: 1,
    operation: "text_to_image",
    body,
    slots: [],
    resource_policy: resourcePolicy,
  };
}

function humanize(name: string): string {
  return name.replaceAll("_", " ");
}

function templateCreateError(error: unknown): string | null {
  if (!error) return null;
  if (error instanceof ApiError && error.code === "prompt-template-name-taken") {
    return "A template with this name already exists. Choose a different name.";
  }
  if (error instanceof ApiError && error.code === "prompt-template-resources-unavailable") {
    return "One or more selected workflows or LoRAs are no longer available.";
  }
  return "That template could not be saved. Review its name and image setup, then try again.";
}

function inputSlots(template: PromptTemplateDetail): InputSlot[] {
  return template.current_revision.contract_json.slots.filter(
    (slot): slot is InputSlot => slot.mode === "input",
  );
}

function initialInputs(template: PromptTemplateDetail, count: number): Record<string, string[]> {
  return Object.fromEntries(inputSlots(template).map((slot) => [
    slot.name,
    Array.from({ length: slot.variation_scope === "item" ? count : 1 }, () => ""),
  ]));
}

function AttemptStatus({
  attempt,
  onClose,
  onRetry,
  onDiscard,
}: {
  attempt: PromptDirectQueueAttempt;
  onClose: () => void;
  onRetry: () => void;
  onDiscard: () => void;
}) {
  const count = attempt.createPayload.item_count;
  const plural = count === 1 ? "prompt" : "prompts";
  const pending = attempt.status === "creating" || attempt.status === "queueing";
  const errorMessage = attempt.errorStage === "admission"
    ? "The generated prompts could not be verified, so nothing new was queued."
    : attempt.errorCode === "prompt-batch-distinct-capacity-exceeded"
      ? "Distinct choice mode cannot create that many prompts. Request fewer prompts, add more choices, or allow repeats."
      : attempt.errorCode === "prompt-model-profile-unset"
        ? "Choose a chat model for this chat before using model-guided template slots."
      : attempt.errorCode === "prompt-model-worker-unavailable"
        ? "The chat model could not be made ready. Check it in Settings, or use authored inputs and choices instead."
        : attempt.errorCode === "prompt-model-invocation-failed"
          ? "The chat model could not fill the template slots. Retry, or use authored inputs and choices instead."
          : attempt.errorCode === "prompt-template-stale"
            ? "This template changed. Start over to use its current revision."
    : attempt.errorStage === "queue"
      ? `The ${plural} were created, but could not be queued. Retry to continue the same safe attempt.`
      : `The ${plural} could not be created. Retry to continue the same safe attempt.`;

  return (
    <section
      className="prompt-template-direct-status"
      aria-busy={pending}
      aria-live="polite"
    >
      <BookOpen size={24} aria-hidden="true" />
      <div>
        <small>{attempt.template.name}</small>
        <h3>
          {attempt.status === "creating"
            ? `Creating ${count} ${plural}...`
            : attempt.status === "queueing"
              ? `Adding ${count} ${plural} to the queue...`
              : attempt.status === "queued"
                ? `${count} ${plural} queued`
                : "Prompt creation needs attention"}
        </h3>
        {pending && <p>You can close this window. The request will continue safely.</p>}
        {attempt.status === "queued" && (
          <p>Generation is queued in this chat. Your composer and generation settings were left unchanged.</p>
        )}
        {attempt.status === "error" && <ErrorCallout message={errorMessage} />}
      </div>
      <footer>
        {pending && <button type="button" className="secondary" onClick={onClose}>Hide</button>}
        {attempt.status === "queued" && (
          <button type="button" className="primary" onClick={onClose}>Done</button>
        )}
        {attempt.status === "error" && (
          <>
            <button type="button" className="secondary" onClick={onClose}>Close</button>
            <button type="button" className="secondary" onClick={onDiscard}>Start over</button>
            <button type="button" className="primary" onClick={onRetry}>
              {attempt.errorStage === "queue" ? "Retry queue" : "Retry"}
            </button>
          </>
        )}
      </footer>
    </section>
  );
}

export function PromptTemplatesDialog({
  currentPrompt,
  maximum,
  attempt,
  onClose,
  onCreate,
  onRetry,
  onDiscard,
}: {
  currentPrompt: string;
  maximum: number;
  attempt: PromptDirectQueueAttempt | null;
  onClose: () => void;
  onCreate: (request: PromptDirectQueueRequest) => void;
  onRetry: () => void;
  onDiscard: () => void;
}) {
  const client = useQueryClient();
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(
    attempt?.template.id ?? null,
  );
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [resourcePolicy, setResourcePolicy] = useState<
    Extract<PromptTemplateResourcePolicy, { mode: "inherited" | "fixed" }>
  >({ mode: "inherited" });
  const [configuration, setConfiguration] = useState<{
    revisionId: string;
    count: number;
    inputs: Record<string, string[]>;
  } | null>(null);
  const templates = useQuery({
    queryKey: ["prompt-templates", false, TEMPLATE_LIMIT, 0],
    queryFn: () => api.promptTemplates(false, TEMPLATE_LIMIT, 0),
    enabled: !attempt,
  });
  const selected = useQuery({
    queryKey: ["prompt-template", selectedId],
    queryFn: () => api.promptTemplate(selectedId!),
    enabled: Boolean(selectedId) && !attempt,
  });
  const create = useMutation({
    mutationFn: () => api.createPromptTemplate({
      idempotency_key: crypto.randomUUID(),
      name: name.trim(),
      description: description.trim(),
      contract: quickTemplateContract(currentPrompt.trim(), resourcePolicy),
    }),
    onSuccess: (created) => {
      client.setQueryData(["prompt-template", created.template.id], created.template);
      void client.invalidateQueries({ queryKey: ["prompt-templates"] });
      setSelectedId(created.template.id);
      setConfiguration(null);
      setCreating(false);
      setName("");
      setDescription("");
      setResourcePolicy({ mode: "inherited" });
    },
  });
  const filtered = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    if (!query) return templates.data?.items ?? [];
    return (templates.data?.items ?? []).filter((template) =>
      `${template.name}\n${template.description}`.toLocaleLowerCase().includes(query),
    );
  }, [search, templates.data?.items]);
  const canSave = Boolean(currentPrompt.trim() && name.trim())
    && promptTemplateImageSetupIsComplete(resourcePolicy)
    && !create.isPending;
  const authoredSlots = selected.data ? inputSlots(selected.data) : [];
  const modelSlots = selected.data?.current_revision.contract_json.slots.filter(
    (slot) => slot.mode === "model",
  ) ?? [];
  const activeConfiguration = selected.data
    ? configuration?.revisionId === selected.data.current_revision.id
      ? configuration
      : {
          revisionId: selected.data.current_revision.id,
          count: 1,
          inputs: initialInputs(selected.data, 1),
        }
    : null;
  const count = activeConfiguration?.count ?? 1;
  const inputs = activeConfiguration?.inputs ?? {};
  const canCreate = Boolean(selected.data && activeConfiguration)
    && Number.isInteger(count)
    && count >= 1
    && count <= maximum
    && Object.values(inputs).every((values) =>
      values.every((value) => value.trim() && value.length <= MAX_INPUT_CHARACTERS));

  const updateCount = (next: number) => {
    if (
      !selected.data
      || !activeConfiguration
      || !Number.isInteger(next)
      || next < 1
      || next > maximum
    ) return;
    setConfiguration({
      revisionId: selected.data.current_revision.id,
      count: next,
      inputs: Object.fromEntries(authoredSlots.map((slot) => {
      const length = slot.variation_scope === "item" ? next : 1;
      const previous = activeConfiguration.inputs[slot.name] ?? [];
      return [slot.name, Array.from({ length }, (_, index) => previous[index] ?? "")];
      })),
    });
  };

  const updateInput = (slot: InputSlot, index: number, value: string) => {
    if (!selected.data || !activeConfiguration) return;
    setConfiguration({
      revisionId: selected.data.current_revision.id,
      count: activeConfiguration.count,
      inputs: {
        ...activeConfiguration.inputs,
        [slot.name]: (activeConfiguration.inputs[slot.name] ?? []).map((entry, currentIndex) =>
        currentIndex === index ? value : entry),
      },
    });
  };

  const submit = () => {
    if (!selected.data || !canCreate) return;
    const serializedInputs: Record<string, string | string[]> = {};
    for (const slot of authoredSlots) {
      serializedInputs[slot.name] = slot.variation_scope === "item"
        ? inputs[slot.name].slice(0, count)
        : inputs[slot.name][0];
    }
    onCreate({
      template: structuredClone(selected.data),
      itemCount: count,
      inputs: serializedInputs,
    });
  };

  return (
    <AccessibleDialog
      title="Prompt templates"
      eyebrow="Create in this chat"
      closeLabel="Close prompt templates"
      className="prompt-templates-dialog"
      onClose={onClose}
    >
      {attempt ? (
        <AttemptStatus
          attempt={attempt}
          onClose={onClose}
          onRetry={onRetry}
          onDiscard={onDiscard}
        />
      ) : (
        <>
          <div className="prompt-templates-intro">
            <p>Create and queue one or more prompts without leaving this chat.</p>
            <button
              type="button"
              className="secondary"
              disabled={!currentPrompt.trim()}
              onClick={() => setCreating((value) => !value)}
            >
              <Plus size={15} /> Save current prompt as a template
            </button>
          </div>

          {creating && (
            <section className="prompt-template-quick-create" aria-labelledby="quick-template-heading">
              <div>
                <h3 id="quick-template-heading">Save this prompt</h3>
                <p>Name this reusable prompt and optionally choose its image setup here.</p>
              </div>
              <label>
                Template name
                <input
                  value={name}
                  maxLength={120}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Product photo"
                />
              </label>
              <label>
                Description <small>(optional)</small>
                <input
                  value={description}
                  maxLength={500}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="When I want a clean studio product shot"
                />
              </label>
              <div className="prompt-template-quick-preview">
                <small>Prompt</small>
                <p>{currentPrompt.trim()}</p>
              </div>
              <PromptTemplateImageSetupPicker value={resourcePolicy} onChange={setResourcePolicy} />
              {!promptTemplateImageSetupIsComplete(resourcePolicy) && (
                <p className="muted">Choose a ready workflow and any installed LoRAs you want to save.</p>
              )}
              <ErrorCallout
                message={templateCreateError(create.error)}
              />
              <div className="prompt-template-quick-actions">
                <button type="button" className="secondary" onClick={() => setCreating(false)}>
                  Cancel
                </button>
                <button type="button" className="primary" disabled={!canSave} onClick={() => create.mutate()}>
                  {create.isPending ? "Saving..." : "Save template"}
                </button>
              </div>
            </section>
          )}

          <label className="prompt-template-search">
            <Search size={15} aria-hidden="true" />
            <span className="sr-only">Search templates</span>
            <input
              aria-label="Search templates"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search templates"
            />
          </label>
          <ErrorCallout
            message={templates.isError ? "Templates could not be loaded. Try again." : null}
            action={templates.isError
              ? <button type="button" className="secondary compact-button" onClick={() => void templates.refetch()}>Retry</button>
              : undefined}
          />
          {templates.isPending && <div className="loading-line" />}
          {!templates.isPending && !filtered.length ? (
            <EmptyState
              icon={<BookOpen />}
              title={search.trim() ? "No matching templates" : "No templates yet"}
              body={search.trim()
                ? "Try another search, or save the prompt already in your composer."
                : "Write a prompt in the composer, then save it here for next time."}
            />
          ) : (
            <div className="prompt-template-picker-layout">
              <ul aria-label="Prompt templates">
                {filtered.map((template) => (
                  <li key={template.id}>
                    <button
                      type="button"
                      className={selectedId === template.id ? "selected" : ""}
                      onClick={() => {
                        setSelectedId(template.id);
                        setConfiguration(null);
                      }}
                    >
                      <BookOpen size={16} />
                      <span><strong>{template.name}</strong><small>{template.description || "Saved prompt template"}</small></span>
                    </button>
                  </li>
                ))}
              </ul>
              <section className="prompt-template-picker-detail" aria-live="polite">
                {!selectedId && <p className="muted">Choose a template to configure it.</p>}
                {selected.isPending && selectedId && <div className="loading-line" />}
                {selected.isError && <ErrorCallout message="That template could not be opened. Try another one." />}
                {selected.data && (
                  <>
                    <small>Saved image prompt</small>
                    <h3>{selected.data.name}</h3>
                    {selected.data.description && <p>{selected.data.description}</p>}
                    <div className="prompt-template-quick-preview">
                      <small>Prompt</small>
                      <p>{selected.data.current_revision.contract_json.body}</p>
                    </div>
                    <label className="prompt-template-count">
                      Number of prompts
                      <input
                        aria-label="Number of prompts"
                        type="number"
                        min={1}
                        max={maximum}
                        step={1}
                        value={count}
                        onChange={(event) => updateCount(event.target.valueAsNumber)}
                      />
                      <small>Up to {maximum} can be queued at once.</small>
                    </label>
                    {authoredSlots.length > 0 && (
                      <section className="prompt-expansion-inputs" aria-labelledby="prompt-template-input-heading">
                        <h4 id="prompt-template-input-heading">Prompt details</h4>
                        {authoredSlots.flatMap((slot) => (inputs[slot.name] ?? []).map((value, index) => (
                          <label key={`${slot.name}-${index}`}>
                            {humanize(slot.name)}
                            {slot.variation_scope === "item" ? ` - prompt ${index + 1}` : " - shared"}
                            <textarea
                              aria-label={`${humanize(slot.name)} ${slot.variation_scope === "item" ? `for prompt ${index + 1}` : "shared across prompts"}`}
                              rows={2}
                              maxLength={MAX_INPUT_CHARACTERS}
                              value={value}
                              onChange={(event) => updateInput(slot, index, event.target.value)}
                            />
                          </label>
                        )))}
                      </section>
                    )}
                    {modelSlots.length > 0 && (
                      <section className="prompt-model-guidance" aria-labelledby="prompt-template-model-heading">
                        <h4 id="prompt-template-model-heading">Automatic details</h4>
                        <dl>
                          {modelSlots.map((slot) => (
                            <div key={slot.name}>
                              <dt>{humanize(slot.name)}</dt>
                              <dd>{slot.mode === "model" ? slot.guidance : null}</dd>
                            </div>
                          ))}
                        </dl>
                      </section>
                    )}
                    <button
                      type="button"
                      className="primary"
                      disabled={!canCreate}
                      onClick={submit}
                    >
                      {count === 1 ? "Create prompt" : `Create ${count} prompts`}
                    </button>
                  </>
                )}
              </section>
            </div>
          )}
        </>
      )}
    </AccessibleDialog>
  );
}
