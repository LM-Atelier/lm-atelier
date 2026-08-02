import type { EditTemplate } from "./types";

const SUBJECT_SLOT = "{subject}";
export const MAX_SUBJECT_CHARACTERS = 2_000;

/** Mirror of the server-side splice; the subject extends the instruction and
 * cannot rewrite its frame. */
export function renderTemplateInstruction(template: EditTemplate, subject: string): string {
  const addition = subject.trim().slice(0, MAX_SUBJECT_CHARACTERS);
  if (template.instruction.includes(SUBJECT_SLOT)) {
    return template.instruction.replace(SUBJECT_SLOT, addition ? ` ${addition}` : "");
  }
  return addition ? `${template.instruction} ${addition}`.trim() : template.instruction;
}
