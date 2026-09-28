import { NATIVE_SETTINGS_KEY, nativeWorkflowSettings } from "./nativeWorkflowSettings";

export function NativeWorkflowFixedSettings({ schema }: { schema?: Record<string, unknown> }) {
  if (!schema || !Object.hasOwn(schema, NATIVE_SETTINGS_KEY)) return null;
  const fixed = nativeWorkflowSettings(schema);
  if (fixed === null) return <p className="muted">The workflow's settings metadata could not be read. Import it again to refresh its controls.</p>;
  if (!fixed.length) return null;
  return <div role="group" aria-label="Settings supplied by the workflow">
    {fixed.map((item) => <div className="setting-row" key={JSON.stringify([item.nodeId, item.inputName])}>
      <span><strong>{item.label}</strong><small>{item.explanation}</small></span>
    </div>)}
  </div>;
}
