import { useState } from "react";
import { LoraStackControl } from "./LoraStackControl";
import { videoLengthDelivery } from "./settings";
import type { SettingField } from "./types";

/** Reads a number without writing one down until it is a number.
 *
 * Its own component because it needs state, and a hook cannot live below the
 * early returns in `SettingControl`.
 *
 * The text in the box and the value of the setting are not the same thing while
 * someone is typing. An empty box is the ordinary middle of select-all-and-
 * retype, not a request to store anything, and the settings surfaces persist
 * every change as it happens - so whatever this reports is written immediately,
 * with no save step to correct it. Previously an emptied integer box reported
 * NaN, which reaches the API as null and is refused, and an emptied number box
 * could not be emptied at all: `Number("")` is 0, so it jumped to zero and
 * stored it.
 *
 * A number input hands its handler the SANITIZED value, so every half-typed
 * state - a lone minus sign, a decimal point before its digits, an exponent
 * mid-way - arrives as the empty string, exactly like a cleared box. That is
 * what makes the empty case the whole class rather than a corner of it.
 *
 * So the raw text is held here and nothing is reported until it parses. On blur
 * the draft is dropped and the box shows the value that was actually stored,
 * which is also how a half-typed entry that never parsed corrects itself.
 */
function NumericSettingControl({
  field,
  value,
  onChange,
  fixed,
}: {
  field: SettingField;
  value: unknown;
  onChange: (value: unknown) => void;
  fixed: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const numericValue = Number(value ?? field.default);
  const delivery = videoLengthDelivery(field, numericValue);
  const durationNote = delivery
    ? Math.abs(delivery.deliveredSeconds - numericValue) > 1e-9
      ? `Requested ${numericValue} seconds · delivers ${delivery.deliveredSeconds} seconds (${delivery.frames} frames).`
      : `Delivers ${delivery.deliveredSeconds} seconds (${delivery.frames} frames).`
    : "";
  return (
    <label className="setting-row">
      <span><strong>{field.label}</strong>{field.help && <small>{field.help}</small>}{durationNote && <small>{durationNote}</small>}</span>
      <input
        type="number"
        value={draft ?? String(numericValue)}
        min={field.minimum ?? undefined}
        max={field.maximum ?? undefined}
        step={field.step ?? (field.type === "integer" ? 1 : 0.01)}
        disabled={fixed}
        onChange={(event) => {
          const text = event.target.value;
          setDraft(text);
          const trimmed = text.trim();
          // `Number("")` is 0, which is why the empty case is answered before
          // parsing rather than by it.
          const parsed = trimmed === ""
            ? Number.NaN
            : field.type === "integer"
              ? Number.parseInt(trimmed, 10)
              : Number(trimmed);
          if (Number.isFinite(parsed)) onChange(parsed);
        }}
        onBlur={() => setDraft(null)}
      />
    </label>
  );
}

export function SettingControl({
  field,
  value,
  onChange,
}: {
  field: SettingField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const fixed = field.choices.length === 1;
  if (field.key === "loras") {
    return <LoraStackControl value={value} onChange={onChange} />;
  }
  if (field.type === "boolean") {
    return (
      <label className="setting-row toggle-row">
        <span><strong>{field.label}</strong>{field.help && <small>{field.help}</small>}</span>
        <input type="checkbox" checked={Boolean(value)} disabled={fixed} onChange={(event) => onChange(event.target.checked)} />
      </label>
    );
  }
  if (field.type === "enum") {
    return (
      <label className="setting-row">
        <span><strong>{field.label}</strong>{field.help && <small>{field.help}</small>}</span>
        <select value={String(value ?? "")} disabled={fixed} onChange={(event) => onChange(event.target.value)}>
          {field.choices.map((choice) => <option key={String(choice)}>{String(choice)}</option>)}
        </select>
      </label>
    );
  }
  if (field.type === "number" || field.type === "integer") {
    return <NumericSettingControl field={field} value={value} onChange={onChange} fixed={fixed} />;
  }
  if (field.type === "array" || field.type === "object") {
    return (
      <label className="setting-row">
        <span><strong>{field.label}</strong>{field.help && <small>{field.help}</small>}</span>
        <textarea
          // This one control is uncontrolled - it holds the user's raw text so a
          // half-typed object is not destroyed on every keystroke - so React will
          // not update it from `value`. Keying it on the value remounts it when the
          // value changes from somewhere else, which is what keeps it in step.
          // The surrounding controls must NOT carry that key: they are controlled,
          // they need no remount, and remounting them loses the caret.
          key={JSON.stringify(value ?? field.default)}
          rows={3}
          disabled={fixed}
          defaultValue={JSON.stringify(value ?? field.default, null, 2)}
          onChange={(event) => { event.currentTarget.dataset.edited = "true"; }}
          onBlur={(event) => {
            if (event.currentTarget.dataset.edited !== "true") return;
            try {
              const parsed = JSON.parse(event.target.value) as unknown;
              event.target.setCustomValidity("");
              delete event.currentTarget.dataset.edited;
              onChange(parsed);
            } catch {
              event.target.setCustomValidity("Enter valid JSON");
              event.target.reportValidity();
            }
          }}
        />
      </label>
    );
  }
  return (
    <label className="setting-row">
      <span><strong>{field.label}</strong>{field.help && <small>{field.help}</small>}</span>
      <input value={String(value ?? "")} disabled={fixed} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}
