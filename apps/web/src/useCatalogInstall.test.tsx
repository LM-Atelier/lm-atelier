import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import type { CatalogModel, CatalogPreflight, Job } from "./types";
import { useCatalogInstall } from "./useCatalogInstall";

vi.mock("./api", () => ({ api: { catalogPreflight: vi.fn(), download: vi.fn() } }));
const model = { name: "Landscape", provider: "civitai", remote_id: "202" } as CatalogModel;

function savedOffers(): unknown {
  return JSON.parse(localStorage.getItem("lm-atelier.model-update-downloads") ?? "[]");
}

function show() {
  const client = new QueryClient();
  return renderHook(useCatalogInstall, { wrapper: ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  ) });
}

beforeEach(() => {
  localStorage.clear();
  vi.mocked(api.catalogPreflight).mockResolvedValue({
    can_install: true, remote_id: "202", source_remote_id: "101", revision: "202", checks: [],
    selected_files: ["model.safetensors"], expected_sha256: { "model.safetensors": "a".repeat(64) },
    install_plan: { id: "plan-a", compatibility: "supported" },
  } as unknown as CatalogPreflight);
  vi.mocked(api.download).mockResolvedValue({ id: "download-a", status: "queued" } as Job);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("retains the original install only after confirmed download acceptance, across navigation", async () => {
  const first = show();
  await act(async () => { await first.result.current.prepare.mutateAsync({ model, selectedRole: "image", previousInstallId: "old-install" }); });
  expect(first.result.current.pendingInstall?.previousInstallId).toBe("old-install");
  expect(savedOffers()).toEqual([]);
  expect(api.download).not.toHaveBeenCalled();
  await act(async () => { await first.result.current.confirm.mutateAsync(first.result.current.pendingInstall!); });
  expect(first.result.current.pendingInstall).toBeNull();
  expect(first.result.current.updateDownloads).toEqual([{ jobId: "download-a", previousInstallId: "old-install", modelName: "Landscape" }]);
  first.unmount();
  const restored = show();
  expect(restored.result.current.updateDownloads).toEqual(savedOffers());
  expect(restored.result.current.updateDownloads).toHaveLength(1);
  act(() => restored.result.current.dismissUpdate("download-a"));
  expect(savedOffers()).toEqual([]);
});

it("does not remember a cancelled preflight or an ordinary catalog installation", async () => {
  const hook = show();
  await act(async () => { await hook.result.current.prepare.mutateAsync({ model, selectedRole: "image", previousInstallId: "old-install" }); });
  act(() => hook.result.current.cancel());
  expect(savedOffers()).toEqual([]);
  await act(async () => { await hook.result.current.prepare.mutateAsync({ model, selectedRole: "image" }); });
  await act(async () => { await hook.result.current.confirm.mutateAsync(hook.result.current.pendingInstall!); });
  expect(hook.result.current.updateDownloads).toEqual([]);
});

it("does not remember a rejected download request", async () => {
  vi.mocked(api.download).mockRejectedValue(new Error("Installation was refused."));
  const hook = show();
  await act(async () => { await hook.result.current.prepare.mutateAsync({ model, selectedRole: "image", previousInstallId: "old-install" }); });
  await act(async () => { await expect(hook.result.current.confirm.mutateAsync(hook.result.current.pendingInstall!)).rejects.toThrow("Installation was refused."); });
  expect(savedOffers()).toEqual([]);
  expect(hook.result.current.pendingInstall).not.toBeNull();
});

it("ignores malformed saved offers", () => {
  localStorage.setItem("lm-atelier.model-update-downloads", '{"jobId":"not-an-array"}');
  expect(show().result.current.updateDownloads).toEqual([]);
});
