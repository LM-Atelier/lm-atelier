import { AccessibleDialog } from "./AccessibleDialog";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import type { EngineCapabilities, EngineRole, GenerationPreset, RoutingMode } from "./types";

export function SettingsDrawer({
  open,
  onClose,
  mode,
  role,
  onRole,
  engines,
  values,
  onValues,
  presets,
  presetId,
  onPreset,
  workflowSchema,
  inheritedValues,
  inheritedPresetId,
  profileValues,
  imageEdit,
  imageEditPrompt,
}: {
  open: boolean;
  onClose: () => void;
  mode: RoutingMode;
  // In auto the routing mode names no single role, so the caller supplies the
  // role being edited and the drawer offers the choice; in every other mode
  // the caller passes roleForMode(mode) and the picker stays hidden.
  role: EngineRole;
  onRole: (role: EngineRole) => void;
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
  imageEdit: boolean;
  imageEditPrompt: string;
}) {
  if (!open) return null;
  return (
    <AccessibleDialog
      title={`${role[0].toUpperCase() + role.slice(1)} settings`}
      eyebrow="Chat defaults"
      closeLabel="Close settings"
      onClose={onClose}
      className={mode === "auto" ? "settings-drawer settings-drawer-with-roles" : "settings-drawer"}
      backdropClassName="settings-drawer-backdrop"
    >
      {mode === "auto" && (
        <div className="segmented compact settings-role-tabs" role="group" aria-label="Settings role">
          {(["chat", "image", "video"] as EngineRole[]).map((option) => (
            <button
              key={option}
              type="button"
              className={role === option ? "active" : ""}
              aria-pressed={role === option}
              onClick={() => onRole(option)}
            >
              {option}
            </button>
          ))}
        </div>
      )}
      {/* No key={role} here. The panel derives its engine, presets and
          capability settings from the role prop on every render, so the key
          was resetting exactly one thing: the reader's basic/advanced/expert
          choice, which is a disclosure level and has nothing to do with which
          role is being edited. */}
      <GenerationSettingsPanel
        role={role}
        engines={engines}
        values={values}
        onValues={onValues}
        presets={presets}
        presetId={presetId}
        onPreset={onPreset}
        workflowSchema={workflowSchema}
        inheritedValues={inheritedValues}
        inheritedPresetId={inheritedPresetId}
        profileValues={profileValues}
        imageEdit={imageEdit}
        imageEditPrompt={imageEditPrompt}
        resetLabel="Reset chat overrides"
        onReset={() => onValues({})}
      />
    </AccessibleDialog>
  );
}
