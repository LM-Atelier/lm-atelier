import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { api } from "./api";
import type { ModelAssetInstall, ModelProfile } from "./types";

type Props = {
  kind: "profile" | "lora";
  id: string;
  name: string;
  savedText: string;
  available: boolean;
  busy?: boolean;
};

function SuggestionDialog({ kind, id, name, savedText, onClose }: Props & { onClose: () => void }) {
  const client = useQueryClient();
  const [baseline] = useState(savedText);
  const [suggestion, setSuggestion] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [generating, setGenerating] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const savingNow = useRef(false);
  const changed = savedText !== baseline;
  const queryKey = kind === "profile" ? "profiles" : "model-assets";
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const request = kind === "profile" ? api.suggestProfileUseCase : api.suggestLoraUseCase;
    void request(id, baseline, controller.signal).then((result) => {
      if (active) {
        setSuggestion(result.suggestion);
        setDraft(result.suggestion);
      }
    }).catch((failure: unknown) => {
      if (active) setError(failure instanceof Error ? failure.message : "The suggestion is unavailable.");
    }).finally(() => { if (active) setGenerating(false); });
    return () => { active = false; controller.abort(); };
  }, [kind, id, baseline, attempt]);
  const refresh = () => {
    void client.invalidateQueries({ queryKey: [queryKey] });
    void client.invalidateQueries({ queryKey: ["workflow-families"] });
    void client.invalidateQueries({ queryKey: ["lora-suggestions"] });
  };
  const save = async () => {
    if (savingNow.current || generating || changed || !suggestion || !draft.trim()) return;
    savingNow.current = true;
    setSaving(true);
    setError(null);
    const values = {
      use_case: draft.trim(),
      use_case_derived: draft.trim() === suggestion.trim(),
      expected_use_case: baseline,
    };
    try {
      const updated = kind === "profile"
        ? await api.updateProfile(id, values)
        : await api.updateModelAsset(id, values);
      client.setQueryData<Array<ModelProfile | ModelAssetInstall>>([queryKey], (current) =>
        current?.map((item) => item.id === id ? updated : item) ?? [updated],
      );
      refresh();
      onClose();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The use case could not be saved.");
      refresh();
    } finally {
      savingNow.current = false;
      setSaving(false);
    }
  };
  return (
    <AccessibleDialog title={`Suggested use case for ${name}`} eyebrow="Local model"
      closeLabel="Close use-case suggestion" onClose={() => { if (!savingNow.current) onClose(); }}>
      <p>The local chat model summarizes the saved provider description. Review the suggestion before saving.</p>
      {baseline && <p>Current use case: {baseline}</p>}
      {generating && <p role="status">Generating a suggestion…</p>}
      <ErrorCallout message={error} />
      {changed && <p role="alert">The saved use case changed. Close this suggestion and review the current text.</p>}
      {suggestion !== null && <label>Suggested use case
        <textarea aria-label={`Suggested use case for ${name}`} rows={4} maxLength={1000}
          value={draft} readOnly={saving || changed} onChange={(event) => setDraft(event.target.value)} />
      </label>}
      {suggestion !== null && <p>{draft.trim() === suggestion.trim()
        ? "Saving keeps this labeled as derived from model metadata."
        : "Your edited text will be saved as a manual use case."}</p>}
      <footer>
        <button className="secondary compact-button" aria-disabled={saving}
          onClick={() => { if (!savingNow.current) onClose(); }}>Cancel</button>
        {!suggestion && !generating && <button className="secondary compact-button"
          aria-disabled={changed} onClick={() => {
            if (!changed) { setGenerating(true); setError(null); setAttempt(attempt + 1); }
          }}>Try again</button>}
        <button className="primary compact-button" aria-disabled={saving || generating || changed || !draft.trim()}
          onClick={() => { void save(); }}>{saving ? "Saving…" : "Save use case"}</button>
      </footer>
    </AccessibleDialog>
  );
}

export function UseCaseSuggestion(props: Props) {
  const [open, setOpen] = useState(false);
  if (!props.available) return null;
  return <>
    <button className="secondary compact-button" aria-label={`Suggest use case for ${props.name}`}
      aria-disabled={props.busy || open} onClick={() => { if (!props.busy && !open) setOpen(true); }}>
      Suggest use case
    </button>
    {open && createPortal(<SuggestionDialog key={`${props.kind}:${props.id}`} {...props} onClose={() => setOpen(false)} />, document.body)}
  </>;
}
