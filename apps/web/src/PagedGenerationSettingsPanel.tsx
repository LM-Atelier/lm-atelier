import type { ComponentProps } from "react";
import { ErrorCallout } from "./ErrorCallout";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import { GenerationPresetPicker } from "./GenerationPresetPicker";
import { useGenerationPresetLibrary } from "./useGenerationPresetLibrary";

export function PagedGenerationSettingsPanel(props: Omit<ComponentProps<typeof GenerationSettingsPanel>, "presets">) {
  const { role, presetId, inheritedPresetId, onPreset, editSettings } = props;
  const library = useGenerationPresetLibrary(role, [presetId, inheritedPresetId], !editSettings);
  if (editSettings) return <GenerationSettingsPanel {...props} presets={[]} />;
  const inherited = library.identityRows.find((preset) => preset.id === inheritedPresetId)
    ?? library.identityRows.find((preset) => preset.is_default);
  const picker = <GenerationPresetPicker library={library} label={props.presetLabel ?? `${role} preset`}
    searchLabel={`Search ${role} presets`} value={presetId ?? ""} onChange={(value) => onPreset(value || null)}>
    <option value="">{inherited ? `Inherit · ${inherited.name}` : "Inherit default"}</option>
  </GenerationPresetPicker>;
  const unavailable = library.error ? <ErrorCallout message={library.error.message} action={<button type="button" className="secondary" onClick={library.retry}>Retry preset settings</button>} />
    : library.pending ? <p role="status">Loading preset settings…</p>
      : library.missing ? <ErrorCallout message="A selected or inherited preset is unavailable." action={<button type="button" className="secondary" onClick={library.retry}>Retry preset settings</button>} /> : undefined;
  return <GenerationSettingsPanel {...props} presets={library.identityRows} presetControl={picker} settingsUnavailable={props.settingsUnavailable ?? unavailable} />;
}
