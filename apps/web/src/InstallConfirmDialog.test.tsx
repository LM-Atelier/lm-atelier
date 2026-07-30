import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { InstallConfirmDialog } from "./InstallConfirmDialog";
import type { CatalogPreflight, SystemInfo } from "./types";

const preflight = {
  download_bytes: 25769803776,
  available_disk_bytes: 429496729600,
  estimated_ram_bytes: 27917287424,
  estimated_vram_bytes: 23622320128,
  checks: [],
} as unknown as CatalogPreflight;

const eightGigabyteGpu = {
  devices: [{ id: "cuda:0", name: "Test GPU", kind: "gpu", total_memory_bytes: 8589934592 }],
} as unknown as SystemInfo;

describe("InstallConfirmDialog", () => {
  afterEach(cleanup);

  it("states the cost of the transfer before it starts", () => {
    render(
      <InstallConfirmDialog
        name="Big Model"
        preflight={preflight}
        system={eightGigabyteGpu}
        pending={false}
        onConfirm={() => undefined}
        onCancel={() => undefined}
      />,
    );

    expect(screen.getByText("24 GB")).toBeInTheDocument();
    expect(screen.getByText("400 GB")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download 24 GB" })).toBeInTheDocument();
  });

  it("warns when the model needs more accelerator memory than the machine has", () => {
    render(
      <InstallConfirmDialog
        name="Big Model"
        preflight={preflight}
        system={eightGigabyteGpu}
        pending={false}
        onConfirm={() => undefined}
        onCancel={() => undefined}
      />,
    );

    expect(screen.getByText(/more accelerator memory than this machine/i)).toBeInTheDocument();
  });

  it("does not warn when the machine has enough accelerator memory", () => {
    const roomy = {
      devices: [{ id: "cuda:0", name: "Big GPU", kind: "gpu", total_memory_bytes: 51539607552 }],
    } as unknown as SystemInfo;

    render(
      <InstallConfirmDialog
        name="Big Model"
        preflight={preflight}
        system={roomy}
        pending={false}
        onConfirm={() => undefined}
        onCancel={() => undefined}
      />,
    );

    expect(screen.queryByText(/more accelerator memory/i)).not.toBeInTheDocument();
  });

  it("cancels without starting anything", () => {
    const onCancel = vi.fn();
    const onConfirm = vi.fn();
    render(
      <InstallConfirmDialog
        name="Big Model"
        preflight={preflight}
        system={eightGigabyteGpu}
        pending={false}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalled();
    expect(onConfirm).not.toHaveBeenCalled();
  });
});
