import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Wand2 } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { MAX_SUBJECT_CHARACTERS, renderTemplateInstruction } from "./editTemplateInstruction";

/** Pick a one-click edit for the attached image.
 *
 * Picking fills the composer with the template's complete instruction - the
 * send stays an ordinary edit turn, reviewed and revisable like any other,
 * and the instruction is visible and editable before anything runs.
 */
export function EditingStudio({
  onPick,
  onClose,
}: {
  onPick: (instruction: string) => void;
  onClose: () => void;
}) {
  const templates = useQuery({ queryKey: ["edit-templates"], queryFn: api.editTemplates });
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
      <footer>
        <button className="secondary" onClick={onClose}>Cancel</button>
        <button
          className="primary"
          disabled={!selected}
          onClick={() => selected && onPick(renderTemplateInstruction(selected, subject))}
        >
          Use this edit
        </button>
      </footer>
    </AccessibleDialog>
  );
}
