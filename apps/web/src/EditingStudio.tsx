import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Wand2 } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { MAX_SUBJECT_CHARACTERS, renderTemplateInstruction } from "./editTemplateInstruction";
import type { EditTemplate } from "./types";

/** Pick a one-click edit for the attached image.
 *
 * Picking fills the composer with the template's complete instruction - the
 * send stays an ordinary edit turn, reviewed and revisable like any other,
 * and the instruction is visible and editable before anything runs.
 */
export function EditingStudio({
  onPick,
  onClose,
  currentInstruction = "",
}: {
  onPick: (instruction: string, template: EditTemplate) => void;
  onClose: () => void;
  // The composer's draft, offered for saving as a personal template.
  currentInstruction?: string;
}) {
  const client = useQueryClient();
  const templates = useQuery({ queryKey: ["edit-templates"], queryFn: api.editTemplates });
  const [saveName, setSaveName] = useState("");
  const save = useMutation({
    mutationFn: () => api.createEditTemplate({
      name: saveName.trim(),
      instruction: currentInstruction.trim(),
    }),
    onSuccess: () => {
      setSaveName("");
      void client.invalidateQueries({ queryKey: ["edit-templates"] });
    },
  });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [subject, setSubject] = useState("");
  const selected = templates.data?.find((template) => template.id === selectedId) ?? null;
  return (
    <AccessibleDialog
      title="Editing studio"
      eyebrow="One-click edits"
      closeLabel="Close editing studio"
      onClose={onClose}
      className="editing-studio"
    >
      <p className="setup-intro">
        Pick an edit for the attached image. The full instruction lands in the
        composer, ready to adjust before sending.
      </p>
      {templates.error && <p role="alert">{templates.error.message}</p>}
      <div className="studio-template-grid" role="listbox" aria-label="Edit templates">
        {(templates.data ?? []).map((template) => (
          <button
            key={template.id}
            role="option"
            aria-selected={template.id === selectedId}
            className={`studio-template ${template.id === selectedId ? "selected" : ""}`}
            onClick={() => setSelectedId(template.id)}
          >
            <Wand2 size={15} aria-hidden="true" />
            <strong>{template.name}</strong>
            <small>{template.description}</small>
          </button>
        ))}
      </div>
      <label className="studio-subject">
        <span>Add detail (optional)</span>
        <input
          value={subject}
          maxLength={MAX_SUBJECT_CHARACTERS}
          placeholder="e.g. focus on the harbor in the background"
          onChange={(event) => setSubject(event.target.value)}
        />
      </label>
      {currentInstruction.trim() && (
        <div className="studio-save">
          <span>Keep the composer's current instruction as a template:</span>
          <input
            aria-label="Template name"
            placeholder="Name this edit"
            value={saveName}
            maxLength={200}
            onChange={(event) => setSaveName(event.target.value)}
          />
          <button
            className="secondary compact-button"
            disabled={!saveName.trim() || save.isPending}
            onClick={() => save.mutate()}
          >
            {save.isPending ? "Saving…" : "Save as template"}
          </button>
          {save.error && <small role="alert">{save.error.message}</small>}
        </div>
      )}
      <footer>
        <button className="secondary" onClick={onClose}>Cancel</button>
        <button
          className="primary"
          disabled={!selected}
          onClick={() => selected && onPick(renderTemplateInstruction(selected, subject), selected)}
        >
          Use this edit
        </button>
      </footer>
    </AccessibleDialog>
  );
}
