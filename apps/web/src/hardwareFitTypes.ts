export type HardwareFitStatus = "recommended" | "likely" | "tight" | "unsupported" | "unknown";
export type HardwareFitBasis = "unknown" | "calculated" | "declared" | "measured" | "tested" | "certified";

export interface HardwareFitReason {
  code:
    | "accelerator_backend_missing"
    | "accelerator_memory_below_minimum"
    | "accelerator_memory_busy"
    | "accelerator_memory_declared"
    | "accelerator_memory_estimated"
    | "accelerator_memory_measured"
    | "accelerator_memory_unknown"
    | "accelerator_missing"
    | "architecture_unsupported"
    | "cpu_capabilities_unknown"
    | "cpu_capability_missing"
    | "evidence_stale"
    | "platform_unsupported"
    | "runtime_backend_missing"
    | "system_memory_below_minimum"
    | "system_memory_busy"
    | "system_memory_declared"
    | "system_memory_estimated"
    | "system_memory_measured"
    | "system_memory_unknown";
  severity: "info" | "warning" | "block";
  message: string;
}

export interface HardwareFitAlternative {
  code: "choose_compatible_backend" | "choose_cpu_compatible_variant" | "choose_smaller_variant" | "free_current_memory" | "install_supported_runtime" | "use_safer_settings";
  message: string;
}

export interface HardwareFitResource {
  kind: "system" | "accelerator";
  capacity_bytes: number;
  available_bytes: number | null;
  required_bytes: number;
  status: HardwareFitStatus;
  basis: HardwareFitBasis;
  immediate_pressure: boolean;
}

export interface HardwareFitSetting {
  key: string;
  label: string;
  unit: string;
  minimum: number;
  maximum: number;
  advisory_only: boolean;
  preserves_user_override: boolean;
}

export interface HardwareFitAdvice {
  status: HardwareFitStatus;
  basis: HardwareFitBasis;
  evidence_label: "tested" | "certified" | null;
  reasons: HardwareFitReason[];
  alternatives: HardwareFitAlternative[];
  resources: HardwareFitResource[];
  settings: HardwareFitSetting[];
}
