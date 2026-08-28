import { useState } from "react";
import { SettingControl } from "./SettingControl";
import {
  IMAGE_EDIT_STRENGTH_MODE_KEY,
  calibratedImageEditStrength,
  estimateImageEditStrength,
  resolveImageEditStrengthMode,
  workflowImageEditCalibration,
  type ImageEditStrengthMode,
  type WorkflowImageEditCalibration,
} from "./imageEditStrength";
import {
  resolveCapabilitySettings,
  resolveWorkflowSettings,
  visibilityRank,
  type Visibility,
} from "./settings";
import type { EngineCapabilities, EngineRole, GenerationPreset, SettingField } from "./types";

/** The generation settings panel and the strength control it owns.
 *
 * Moved out of App.tsx as its own change, adding nothing. It came out as a
 * cluster because it is one: the panel is the only caller of the strength
 * control, so moving either alone would have made the import circular.
 */

function ImageEditStrengthControl({
  field,
  parameter,
  calibration,
  resolvedSteps,
  prompt,
  layers,
  numericManualLayers,
  values,
  onValues,
}: {
  field: SettingField;
  parameter: string;
  calibration: WorkflowImageEditCalibration | null;
  resolvedSteps: unknown;
  prompt: string;
  layers: Array<Record<string, unknown> | undefined>;
  numericManualLayers: boolean[];
  values: Record<string, unknown>;
  onValues: (values: Record<string, unknown>) => void;
}) {
  const mode: ImageEditStrengthMode = resolveImageEditStrengthMode(
    parameter,
    layers,
    numericManualLayers,
  );
  const activeCalibration = calibration ? {
    ...calibration,
    minimum: field.minimum ?? calibration.minimum,
    maximum: field.maximum ?? calibration.maximum,
  } : null;
  const estimate = activeCalibration
    ? calibratedImageEditStrength(prompt, activeCalibration, resolvedSteps)
    : estimateImageEditStrength(prompt, field.minimum ?? 0, field.maximum ?? 1);
  let manualValue = estimate.value;
  for (const layer of layers) {
    if (typeof layer?.[parameter] === "number") manualValue = layer[parameter];
  }
  const selectAuto = () => {
    const next: Record<string, unknown> = {
      ...values,
      [IMAGE_EDIT_STRENGTH_MODE_KEY]: "auto",
    };
    delete next[parameter];
    onValues(next);
  };
  const selectManual = () => onValues({
    ...values,
    [IMAGE_EDIT_STRENGTH_MODE_KEY]: "manual",
    [parameter]: typeof values[parameter] === "number" ? values[parameter] : manualValue,
  });
  return (
    <div className="setting-row image-edit-strength-control">
      <span>
        <strong>Change strength</strong>
        <small>{mode === "auto" ? `Predicted: ${estimate.scope}` : "Set for this chat"}</small>
      </span>
      <div className="image-edit-strength-inputs">
        <div className="segmented compact" role="group" aria-label="Image edit change strength mode">
          <button type="button" aria-pressed={mode === "auto"} className={mode === "auto" ? "active" : ""} onClick={selectAuto}>Auto</button>
          <button type="button" aria-pressed={mode === "manual"} className={mode === "manual" ? "active" : ""} onClick={selectManual}>Manual</button>
        </div>
        {mode === "manual" && (
          <input
            aria-label="Manual change strength"
            type="number"
            value={manualValue}
            min={field.minimum ?? undefined}
            max={field.maximum ?? undefined}
            step={field.step ?? 0.01}
            onChange={(event) => onValues({
              ...values,
              [IMAGE_EDIT_STRENGTH_MODE_KEY]: "manual",
              [parameter]: Number(event.target.value),
            })}
          />
        )}
      </div>
    </div>
  );
}

export function GenerationSettingsPanel({
  role,
  engines,
  values,
  onValues,
  presets,
  presetId,
  onPreset,
  workflowSchema,
  inheritedValues = {},
  inheritedPresetId = null,
  profileValues = {},
  imageEdit = false,
  imageEditPrompt = "",
  presetLabel = `${role} preset`,
  resetLabel,
  onReset,
}: {
  role: EngineRole;
  engines: EngineCapabilities[];
  values: Record<string, unknown>;
  onValues: (values: Record<string, unknown>) => void;
  presets: GenerationPreset[];
  presetId: string | null;
  onPreset: (presetId: string | null) => void;
  workflowSchema?: Record<string, unknown>;
  inheritedValues?: Record<string, unknown>;
  inheritedPresetId?: string | null;
  profileValues?: Record<string, unknown>;
  imageEdit?: boolean;
  imageEditPrompt?: string;
  presetLabel?: string;
  resetLabel: string;
  onReset: () => void;
}) {
  const [visibility, setVisibility] = useState<Visibility>("basic");
  const engine = engines.find((item) => item.roles.includes(role));
  const rolePresets = presets.filter((preset) => preset.role === role);
  const defaultPreset = rolePresets.find((preset) => preset.is_default);
  const inheritedPreset = rolePresets.find((preset) => preset.id === inheritedPresetId);
  const selectedPreset = rolePresets.find((preset) => preset.id === presetId);
  const inheritedName = inheritedPreset?.name ?? defaultPreset?.name;
  const allFields = resolveWorkflowSettings(
    resolveCapabilitySettings(engine, role),
    workflowSchema,
  );
  const editCalibration = workflowImageEditCalibration(workflowSchema);
  const strengthParameter = editCalibration?.parameter ?? "denoise";
  const strengthField = allFields.find(
    (field) => field.key === strengthParameter && field.available,
  );
  const visibleFields = allFields.filter(
    (field) =>
      field.scope !== "load"
      && visibilityRank[field.visibility] <= visibilityRank[visibility]
      && field.available
      && field.key !== strengthParameter,
  );
  // LoRAs are a list of assets with their own strengths, not one more number
  // among steps and guidance. They get their own section so choosing one is a
  // deliberate act rather than scrolling past it.
  const loraField = visibleFields.find((field) => field.key === "loras");
  const fields = visibleFields.filter((field) => field.key !== "loras");
  const effectiveValue = (field: SettingField): unknown => {
    let value = field.default;
    for (const layer of [
      profileValues,
      defaultPreset?.settings_json,
      inheritedPreset?.settings_json,
      inheritedValues,
      selectedPreset?.settings_json,
      values,
    ]) {
      if (layer && Object.prototype.hasOwnProperty.call(layer, field.key)) {
        value = layer[field.key];
      }
    }
    return value;
  };
  const stepsField = editCalibration?.stepsParameter
    ? allFields.find((field) => field.key === editCalibration.stepsParameter)
    : undefined;
  const resolvedEditSteps = stepsField ? effectiveValue(stepsField) : undefined;
  return (
    <div className="generation-settings-panel">
      <div className="segmented compact" role="group" aria-label="Settings detail level">
        {(["basic", "advanced", "expert"] as Visibility[]).map((level) => (
          <button
            key={level}
            type="button"
            className={visibility === level ? "active" : ""}
            aria-pressed={visibility === level}
            onClick={() => setVisibility(level)}
          >
            {level}
          </button>
        ))}
      </div>
      <div className="settings-list">
        <label className="setting-row">
          <span><strong>Preset</strong></span>
          <select
            aria-label={presetLabel}
            value={presetId ?? ""}
            onChange={(event) => onPreset(event.target.value || null)}
          >
            <option value="">{inheritedName ? `Inherit · ${inheritedName}` : "Inherit default"}</option>
            {rolePresets.map((preset) => (
              <option key={preset.id} value={preset.id}>{preset.name}</option>
            ))}
          </select>
        </label>
        {imageEdit && strengthField && (
          <ImageEditStrengthControl
            field={strengthField}
            parameter={strengthParameter}
            calibration={editCalibration}
            resolvedSteps={resolvedEditSteps}
            prompt={imageEditPrompt}
            layers={[
              profileValues,
              defaultPreset?.settings_json,
              inheritedPreset?.settings_json,
              inheritedValues,
              selectedPreset?.settings_json,
              values,
            ]}
            numericManualLayers={[false, false, true, true, true, true]}
            values={values}
            onValues={onValues}
          />
        )}
        {fields.map((field) => (
          <SettingControl
            key={`${field.scope}:${field.key}:${JSON.stringify(values[field.key])}`}
            field={field}
            value={effectiveValue(field)}
            onChange={(value) => onValues({ ...values, [field.key]: value })}
          />
        ))}
        {!engine && <p className="muted">No {role} engine is configured.</p>}
      </div>
      {loraField && (
        <section className="settings-section" aria-label="LoRAs">
          <h4>LoRAs</h4>
          <div className="settings-list">
            <SettingControl
              field={loraField}
              value={effectiveValue(loraField)}
              onChange={(value) => onValues({ ...values, [loraField.key]: value })}
            />
          </div>
        </section>
      )}
      <div className="generation-settings-actions">
        <button className="secondary" type="button" onClick={onReset}>{resetLabel}</button>
      </div>
    </div>
  );
}
