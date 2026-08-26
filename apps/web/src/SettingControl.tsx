import { LoraStackControl } from "./LoraStackControl";
import { videoLengthDelivery } from "./settings";
import type { SettingField } from "./types";

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
          value={numericValue}
          min={field.minimum ?? undefined}
          max={field.maximum ?? undefined}
          step={field.step ?? (field.type === "integer" ? 1 : 0.01)}
          disabled={fixed}
          onChange={(event) => onChange(field.type === "integer" ? Number.parseInt(event.target.value) : Number(event.target.value))}
        />
      </label>
    );
  }
  if (field.type === "array" || field.type === "object") {
    return (
      <label className="setting-row">
        <span><strong>{field.label}</strong>{field.help && <small>{field.help}</small>}</span>
        <textarea
          rows={3}
          disabled={fixed}
          defaultValue={JSON.stringify(value ?? field.default, null, 2)}
          onBlur={(event) => {
            try {
              const parsed = JSON.parse(event.target.value) as unknown;
              event.target.setCustomValidity("");
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
