import type { ImageInputRole } from "./types";

/** Choose what a picture contributes without adding another composer toolbar. */
export function ImagePurposeControl({ value, disabled, onChange }: {
  value: ImageInputRole | undefined;
  disabled?: boolean;
  onChange: (role: ImageInputRole | undefined) => void;
}) {
  return <label className="attachment-purpose">
    <span>Use as</span>
    <select aria-label="Picture purpose" value={value ?? "automatic"} aria-disabled={disabled || undefined}
      onChange={(event) => {
        if (disabled) return;
        const role = event.target.value;
        if (role === "automatic") onChange(undefined);
        else if (role === "edit_source" || role === "reference") onChange(role);
      }}>
      <option value="automatic">Automatic</option>
      <option value="edit_source">Picture to edit</option>
      <option value="reference">Reference</option>
    </select>
  </label>;
}
