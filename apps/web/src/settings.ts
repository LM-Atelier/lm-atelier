import type { EngineCapabilities, EngineRole, SettingField } from "./types";

export function resolveCapabilitySettings(
  engine: EngineCapabilities | undefined,
  role: EngineRole,
): SettingField[] {
  return engine?.settings_by_role?.[role] ?? engine?.settings ?? [];
}
