import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, renderHook, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { InstallConfirmDialog } from "./InstallConfirmDialog";
import type { CatalogModel, CatalogPreflight, HardwareFitAdvice } from "./types";
import { useCatalogInstall } from "./useCatalogInstall";

vi.mock("./api", () => ({ api: { catalogPreflight: vi.fn(), download: vi.fn() } }));
afterEach(() => { cleanup(); vi.resetAllMocks(); });
const fit: HardwareFitAdvice = { status: "likely", basis: "calculated", evidence_label: null, reasons: [], resources: [], settings: [], alternatives: [] };
const preflight: CatalogPreflight = {
  remote_id: "example/model", revision: "a".repeat(40), selected_files: ["chosen.gguf"],
  expected_sha256: { "chosen.gguf": "b".repeat(64) }, can_install: true, checks: [],
  download_bytes: 2 ** 30, download_size_complete: true, available_disk_bytes: 10 * 2 ** 30,
  source_remote_id: null, comfy_paths: {}, workflow_template_id: null, workflow_template_sha256: null,
  estimated_ram_bytes: null, estimated_vram_bytes: null,
  hardware_fit: fit, install_plan: { id: "original-plan", compatibility: "supported", plan_hash: "original", family: null, failure_code: null, failure_reason: null },
  hardware_alternatives: [{ selected_files: ["split-00001-of-00002.gguf", "split-00002-of-00002.gguf"],
    download_bytes: 2 ** 29, download_size_complete: true, hardware_fit: fit }],
};
const model = { name: "Fixture", provider: "huggingface", remote_id: "example/model" } as CatalogModel;
function show() {
  const client = new QueryClient();
  vi.mocked(api.catalogPreflight).mockResolvedValue(preflight);
  return renderHook(useCatalogInstall, { wrapper: ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  ) });
}

it("reviews an explicit complete variant and blocks download while checking it without losing focus", () => {
  const choose = vi.fn(), confirm = vi.fn();
  const props = { name: "Fixture", preflight, pending: false, onConfirm: confirm, onCancel: vi.fn(), onSelectAlternative: choose };
  const view = render(<InstallConfirmDialog {...props} />);
  expect(screen.getByText("Selected files: chosen.gguf")).toBeInTheDocument();
  expect(screen.getByText(/calculated estimates, not tested results/)).toBeInTheDocument();
  const button = screen.getByRole("button", { name: "Review split-00001-of-00002.gguf" });
  button.focus();
  fireEvent.click(button);
  expect(choose).toHaveBeenCalledWith(preflight.hardware_alternatives![0].selected_files);
  expect(confirm).not.toHaveBeenCalled();
  view.rerender(<InstallConfirmDialog {...props} selectingAlternative />);
  expect(button).toHaveFocus();
  fireEvent.click(button);
  fireEvent.click(screen.getByRole("button", { name: "Checking…" }));
  expect(choose).toHaveBeenCalledOnce();
  expect(confirm).not.toHaveBeenCalled();
});

it("uses a fresh pinned plan and its hashes only after explicit alternative review", async () => {
  const hook = show();
  await act(async () => { await hook.result.current.prepare.mutateAsync({ model, selectedRole: "chat", previousInstallId: "old" }); });
  expect(api.catalogPreflight).toHaveBeenCalledTimes(1);
  expect(api.download).not.toHaveBeenCalled();
  const files = preflight.hardware_alternatives![0].selected_files;
  const fresh = { ...preflight, selected_files: files, expected_sha256: Object.fromEntries(files.map((name) => [name, "c".repeat(64)])), install_plan: { ...preflight.install_plan!, id: "new-plan" } };
  vi.mocked(api.catalogPreflight).mockResolvedValue(fresh);
  await act(async () => { await hook.result.current.selectAlternative.mutateAsync({ pending: hook.result.current.pendingInstall!, files }); });
  expect(api.catalogPreflight).toHaveBeenLastCalledWith("example/model", "chat", "llama.cpp", "a".repeat(40), files, null, null, "huggingface");
  expect(hook.result.current.pendingInstall?.preflight).toBe(fresh);
  expect(hook.result.current.pendingInstall?.previousInstallId).toBe("old");
  expect(api.download).not.toHaveBeenCalled();
  vi.mocked(api.download).mockResolvedValue({ id: "job" } as Awaited<ReturnType<typeof api.download>>);
  await act(async () => { await hook.result.current.confirm.mutateAsync(hook.result.current.pendingInstall!); });
  expect(api.download).toHaveBeenCalledWith("example/model", null, "chat", "llama.cpp", "a".repeat(40), files, fresh.expected_sha256, {}, {}, null, null, "new-plan", null, "unknown");
});

it("retains the reviewed selection after refusal and does not reopen a cancelled dialog on late success", async () => {
  const hook = show();
  await act(async () => { await hook.result.current.prepare.mutateAsync({ model, selectedRole: "chat" }); });
  const pending = hook.result.current.pendingInstall!;
  const files = preflight.hardware_alternatives![0].selected_files;
  vi.mocked(api.catalogPreflight).mockResolvedValue({ ...preflight, can_install: false, checks: [{ id: "disk", label: "Space", status: "block", detail: "Insufficient disk space." }] });
  await act(async () => { await expect(hook.result.current.selectAlternative.mutateAsync({ pending, files })).rejects.toThrow("Insufficient disk space."); });
  expect(hook.result.current.pendingInstall).toBe(pending);
  let finish!: (value: CatalogPreflight) => void;
  vi.mocked(api.catalogPreflight).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  let work!: Promise<unknown>;
  await act(async () => { work = hook.result.current.selectAlternative.mutateAsync({ pending, files }); });
  act(() => hook.result.current.cancel());
  await act(async () => { finish({ ...preflight, selected_files: files }); await work; });
  expect(hook.result.current.pendingInstall).toBeNull();
  expect(api.download).not.toHaveBeenCalled();
});
