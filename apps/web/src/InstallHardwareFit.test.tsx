import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { InstallConfirmDialog } from "./InstallConfirmDialog";
import type { CatalogPreflight, HardwareFitAdvice } from "./types";

const gib = 1024 ** 3;
const fit: HardwareFitAdvice = {
  status: "likely",
  basis: "calculated",
  evidence_label: null,
  reasons: [{ code: "accelerator_memory_busy", severity: "warning", message: "Only 1 GB is free now." }],
  alternatives: [{ code: "free_current_memory", message: "Unload idle models before starting." }],
  resources: [{ kind: "accelerator", capacity_bytes: 16 * gib, available_bytes: gib, required_bytes: 4 * gib, status: "likely", basis: "calculated", immediate_pressure: true }],
  settings: [],
};
const preflight: CatalogPreflight = {
  remote_id: "example/model",
  source_remote_id: null,
  revision: "a".repeat(40),
  selected_files: ["selected.safetensors"],
  expected_sha256: { "selected.safetensors": "b".repeat(64) },
  comfy_paths: {},
  workflow_template_id: null,
  workflow_template_sha256: null,
  download_bytes: 2 * gib,
  available_disk_bytes: 100 * gib,
  estimated_ram_bytes: 3 * gib,
  estimated_vram_bytes: 4 * gib,
  can_install: true,
  install_plan: null,
  checks: [{ id: "memory-availability", label: "Memory available now", status: "warn", detail: "Only 1 GB is free now." }],
  hardware_fit: fit,
};

function show(advice: HardwareFitAdvice = fit) {
  const onConfirm = vi.fn();
  render(<InstallConfirmDialog name="Fixture model" preflight={{ ...preflight, hardware_fit: advice }} pending={false} onConfirm={onConfirm} onCancel={vi.fn()} />);
  return onConfirm;
}

describe("Install hardware advice", () => {
  afterEach(cleanup);

  it("separates estimated fit from temporary pressure without duplicating warnings", () => {
    show();
    expect(screen.getByRole("heading", { name: "Likely to fit" })).toBeInTheDocument();
    expect(screen.getByText("Calculated estimate, not a tested result.")).toBeInTheDocument();
    expect(screen.getByText(/4.0 GB needed.*16 GB total/)).toBeInTheDocument();
    expect(screen.getByText("1.0 GB free now")).toBeInTheDocument();
    expect(screen.getAllByText("Only 1 GB is free now.")).toHaveLength(1);
    expect(screen.getByText("Unload idle models before starting.")).toBeInTheDocument();
  });

  it("keeps unknown fit installable and does not invent free memory", () => {
    const onConfirm = show({ ...fit, status: "unknown", basis: "unknown", reasons: [], alternatives: [], resources: [{ ...fit.resources[0], capacity_bytes: 0, available_bytes: null }] });
    expect(screen.getByRole("heading", { name: "Hardware fit unknown" })).toBeInTheDocument();
    expect(screen.getByText(/capacity unknown/i)).toBeInTheDocument();
    expect(screen.getByText("Free memory not reported")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Download 2.0 GB" }));
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it("shows bounded suggestions while keeping an explicit setting", () => {
    const onConfirm = show({ ...fit, settings: [{ key: "context_tokens", label: "Context", unit: "tokens", minimum: 2048, maximum: 4096, advisory_only: true, preserves_user_override: true }] });
    expect(screen.getByText("Context")).toBeInTheDocument();
    expect(screen.getByText("Suggested range: 2048–4096 tokens")).toBeInTheDocument();
    expect(screen.getByText("Your chosen setting will be kept.")).toBeInTheDocument();
    expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it.each(["tested", "certified"] as const)("shows %s only with the matching evidence basis", (label) => {
    show({ ...fit, basis: label, evidence_label: label });
    expect(screen.getByText(label === "tested" ? "Tested on this exact setup." : "Certified for this exact setup.")).toBeInTheDocument();
  });

  it("does not elevate a calculated result with an inconsistent evidence label", () => {
    show({ ...fit, evidence_label: "tested" });
    expect(screen.getByText("Calculated estimate, not a tested result.")).toBeInTheDocument();
    expect(screen.queryByText("Tested on this exact setup.")).not.toBeInTheDocument();
  });
});
