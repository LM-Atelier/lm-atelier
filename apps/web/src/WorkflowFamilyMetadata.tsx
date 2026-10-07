import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import type { WorkflowFamily } from "./types";
import "./WorkflowFamilyMetadata.css";

export function WorkflowFamilyMetadata({ family }: { family: WorkflowFamily }) {
  const client = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(family.name);
  const [useCase, setUseCase] = useState(family.use_case);
  const [useCaseEdited, setUseCaseEdited] = useState(false);
  const [saved, setSaved] = useState(false);
  const nameInput = useRef<HTMLInputElement>(null);
  const editButton = useRef<HTMLButtonElement>(null);
  const opened = useRef(false);
  useEffect(() => {
    if (editing) {
      opened.current = true;
      nameInput.current?.focus();
    } else if (opened.current) {
      opened.current = false;
      editButton.current?.focus();
    }
  }, [editing]);
  // Compatibility families also update the profile that can be exported.
  const nameLimit = family.compatibility ? 200 : 240;
  const useCaseLimit = family.compatibility ? 1000 : 10000;
  const valid = name.trim().length > 0 && name.trim().length <= nameLimit
    && useCase.trim().length <= useCaseLimit;
  const save = useMutation({
    mutationFn: ({ familyId, values }: { familyId: string; values: { name: string; use_case?: string } }) =>
      api.updateWorkflowFamily(familyId, values),
    onSuccess: (updated, { familyId }) => {
      client.setQueriesData<WorkflowFamily[]>({ queryKey: ["workflow-families"],
        predicate: (query) => Array.isArray(query.state.data)
          && query.queryKey[1] !== "operations" },
        (families) => families?.map((one) => one.id === familyId ? updated : one));
      client.setQueryData(["workflow-family", familyId], updated);
      void client.invalidateQueries({ queryKey: ["workflow-families"] });
      void client.invalidateQueries({ queryKey: ["workflow-family", familyId] });
      void client.invalidateQueries({ queryKey: ["workflows"] });
      void client.invalidateQueries({ queryKey: ["profiles"] });
      void client.invalidateQueries({ queryKey: ["studio-capabilities"] });
      setEditing(false);
      setSaved(true);
    },
  });

  return (
    <section className="workflow-family-metadata" aria-label="Family details">
      {editing ? (
        <form onSubmit={(event) => {
          event.preventDefault();
          if (valid && !save.isPending) {
            save.mutate({ familyId: family.id, values: {
              name: name.trim(), ...(useCaseEdited ? { use_case: useCase.trim() } : {}),
            } });
          }
        }}>
          <label>Family name
            <input ref={nameInput} value={name} onChange={(event) => setName(event.target.value)}
              maxLength={nameLimit} required readOnly={save.isPending} />
          </label>
          <label>Use case
            <textarea value={useCase} onChange={(event) => { setUseCase(event.target.value); setUseCaseEdited(true); }}
              maxLength={useCaseLimit} rows={3} readOnly={save.isPending} />
          </label>
          {family.use_case_derived && !useCaseEdited && <p className="muted">Derived from model metadata</p>}
          <p className="muted">Describe when this family is useful. Automatic workflow selection uses this description.</p>
          {(name.trim().length > nameLimit || useCase.trim().length > useCaseLimit) && (
            <p role="alert">Use up to {nameLimit} characters for the name and {useCaseLimit} for the use case.</p>
          )}
          {save.error && <ErrorCallout message={save.error.message} />}
          <div className="row-actions">
            <button type="submit" className="secondary compact-button" aria-disabled={!valid || save.isPending}>
              {save.isPending ? "Saving family details…" : "Save family details"}
            </button>
            <button type="button" className="secondary compact-button" aria-disabled={save.isPending}
              onClick={() => { if (!save.isPending) { setEditing(false); save.reset(); } }}>
              Cancel family changes
            </button>
          </div>
        </form>
      ) : (
        <>
          <div className="workflow-family-metadata-header">
            <h3>{family.name}</h3>
            <button ref={editButton} className="secondary compact-button" onClick={() => {
              setName(family.name);
              setUseCase(family.use_case);
              setUseCaseEdited(false);
              setSaved(false);
              save.reset();
              setEditing(true);
            }}>Edit family details</button>
          </div>
          <p className="muted">{family.use_case || "No use case has been added."}</p>
          {family.use_case_derived && <p className="muted">Derived from model metadata</p>}
          {saved && <p role="status">Family details saved.</p>}
        </>
      )}
    </section>
  );
}
