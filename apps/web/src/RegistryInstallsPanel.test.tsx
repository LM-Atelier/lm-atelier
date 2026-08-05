import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { api } from "./api";
import { RegistryInstallsPanel } from "./RegistryInstallsPanel";
import type { RegistryInstall, RegistryInstallReview } from "./types";

vi.mock("./api", () => ({
  api: {
    registryInstalls: vi.fn(),
    reviewRegistryInstall: vi.fn(),
    activateRegistryInstall: vi.fn(),
    deactivateRegistryInstall: vi.fn(),
    renewRegistryInstall: vi.fn(),
  },
}));

const install = (overrides: Partial<RegistryInstall> = {}): RegistryInstall => ({
  id: "install-1",
  package_id: "comfyui-example-node",
  package_version: "1.2.3",
  node_types: ["ExampleNode"],
  archive_sha256: "a".repeat(64),
  manifest_sha256: "b".repeat(64),
  wheel_closure_sha256: null,
  wheel_environment_sha256: null,
  trusted: false,
  active: false,
  reviewed_at: null,
  activated_at: null,
  review: null,
  ...overrides,
});

const review = (overrides: Partial<RegistryInstallReview> = {}): RegistryInstallReview => ({
  file_count: 12,
  expanded_bytes: 2048,
  python_file_count: 4,
  install_scripts: [],
  startup_hooks: [],
  native_files: [],
  dependency_manifests: [],
  top_level_entries: ["example-node"],
  registry_warnings: [],
  ...overrides,
});

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <RegistryInstallsPanel />
    </QueryClientProvider>,
  );
}

describe("RegistryInstallsPanel", () => {
  beforeEach(() => {
    vi.mocked(api.registryInstalls).mockResolvedValue([install()]);
    vi.mocked(api.reviewRegistryInstall).mockResolvedValue(install({ trusted: true }));
    vi.mocked(api.activateRegistryInstall).mockResolvedValue(install({ active: true }));
    vi.mocked(api.deactivateRegistryInstall).mockResolvedValue(install());
    vi.mocked(api.renewRegistryInstall).mockResolvedValue({ id: "job-renew" } as never);
  });
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("stays out of the way until something is prepared", async () => {
    vi.mocked(api.registryInstalls).mockResolvedValue([]);
    const { container } = renderPanel();
    await Promise.resolve();
    expect(container.querySelector("section")).toBeNull();
  });

  it("presents granting trust as a deliberate confirmed decision", async () => {
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Trust package" }));

    // The question names what is being trusted and what trusting permits -
    // none of which an OS confirmation could carry.
    expect(screen.getByText(/run inside ComfyUI on this machine/)).toBeInTheDocument();
    expect(screen.getByText("a".repeat(64))).toBeInTheDocument();
    expect(api.reviewRegistryInstall).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "I reviewed this package - trust it" }));
    await waitFor(() => expect(api.reviewRegistryInstall).toHaveBeenCalledWith("install-1", true));
  });

  it("shows what there is to review before asking whether it was reviewed", async () => {
    vi.mocked(api.registryInstalls).mockResolvedValue([
      install({
        review: review({
          install_scripts: ["example-node/install.py"],
          startup_hooks: ["example-node/prestartup_script.py"],
          native_files: ["example-node/_fast.pyd"],
          registry_warnings: ["source_review_required"],
        }),
      }),
    ]);
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Trust package" }));

    // Confirming "I reviewed this package" is only meaningful if the things
    // that decide the answer were on screen.
    expect(screen.getByText("example-node/install.py")).toBeInTheDocument();
    expect(screen.getByText("example-node/prestartup_script.py")).toBeInTheDocument();
    expect(screen.getByText("example-node/_fast.pyd")).toBeInTheDocument();
    expect(screen.getByText(/source review required/)).toBeInTheDocument();
    expect(screen.getByText(/12 files/)).toBeInTheDocument();
  });

  it("says a clean package is clean rather than listing empty rows", async () => {
    vi.mocked(api.registryInstalls).mockResolvedValue([install({ review: review() })]);
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Trust package" }));

    expect(
      screen.getByText("No install scripts, startup hooks, or compiled binaries."),
    ).toBeInTheDocument();
  });

  it("says nothing about findings for a package prepared before they were recorded", async () => {
    // Absent is not the same claim as empty: one means nothing was found, the
    // other means nobody looked.
    vi.mocked(api.registryInstalls).mockResolvedValue([install({ review: null })]);
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Trust package" }));

    expect(screen.queryByText(/No install scripts/)).not.toBeInTheDocument();
    expect(screen.queryByText("Contents")).not.toBeInTheDocument();
  });

  it("does nothing when the trust confirmation is declined", async () => {
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Trust package" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(api.reviewRegistryInstall).not.toHaveBeenCalled();
  });

  it("revokes trust without a confirmation gate", async () => {
    vi.mocked(api.registryInstalls).mockResolvedValue([install({ trusted: true })]);
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Revoke trust" }));

    // Withdrawing permission is not the dangerous direction, so it is not gated.
    await waitFor(() => expect(api.reviewRegistryInstall).toHaveBeenCalledWith("install-1", false));
  });

  it("offers activation only after trust, and deactivation only while active", async () => {
    vi.mocked(api.registryInstalls).mockResolvedValue([install({ trusted: true })]);
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Activate" }));
    await waitFor(() => expect(api.activateRegistryInstall).toHaveBeenCalledWith("install-1"));
    expect(screen.queryByRole("button", { name: "Deactivate" })).toBeNull();

    cleanup();
    vi.mocked(api.registryInstalls).mockResolvedValue([
      install({ trusted: true, active: true }),
    ]);
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Deactivate" }));
    await waitFor(() => expect(api.deactivateRegistryInstall).toHaveBeenCalledWith("install-1"));
    expect(screen.queryByRole("button", { name: "Activate" })).toBeNull();
  });

  it("refreshes dependencies only while the package is inactive", async () => {
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Refresh dependencies" }));
    await waitFor(() => expect(api.renewRegistryInstall).toHaveBeenCalledWith("install-1"));

    cleanup();
    vi.mocked(api.registryInstalls).mockResolvedValue([install({ trusted: true, active: true })]);
    renderPanel();
    expect(await screen.findByText("comfyui-example-node")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh dependencies" })).toBeNull();
  });

  it("speaks the server's stable refusal codes in plain words", async () => {
    vi.mocked(api.reviewRegistryInstall).mockRejectedValue(
      Object.assign(new Error("409 Conflict"), { code: "media_worker_running" }),
    );
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Trust package" }));
    fireEvent.click(screen.getByRole("button", { name: "I reviewed this package - trust it" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Stop the media worker before changing package trust or activation",
    );
  });
});
