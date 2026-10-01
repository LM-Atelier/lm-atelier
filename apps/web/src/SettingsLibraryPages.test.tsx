import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { SettingsView } from "./SettingsView";
import { useAppearance } from "./theme";
import type { GenerationPreset, ModelProfile } from "./types";

const pages = vi.hoisted(() => ({ profile: vi.fn(), preset: vi.fn() }));
vi.mock("./api", () => ({ api: {
  system: vi.fn().mockResolvedValue(null), about: vi.fn().mockResolvedValue(null),
  profiles: vi.fn(), presets: vi.fn(), profilesPage: pages.profile, presetsPage: pages.preset,
  workers: vi.fn().mockResolvedValue([]), runtimes: vi.fn().mockResolvedValue([]), backups: vi.fn().mockResolvedValue([]),
  updateProfile: vi.fn(), clonePreset: vi.fn(), importProfile: vi.fn(), importPreset: vi.fn(),
} }));
vi.mock("./OutputShapeSettings", () => ({ OutputShapeSettings: () => null }));
vi.mock("./SettingDetailSetting", () => ({ SettingDetailSetting: () => null }));

const profile: ModelProfile = {
  id: "profile", name: "Neutral profile", role: "chat", engine: "mock", model_install_id: null,
  use_case: "", load_settings_json: {}, request_settings_json: {}, is_default: false,
};
const preset: GenerationPreset = { id: "preset", name: "Neutral preset", role: "chat", settings_json: {}, is_default: false };
const kinds = ["profile", "preset"] as const;
type Kind = typeof kinds[number];
const label = (kind: Kind) => kind === "profile" ? "model profiles" : "generation presets";
const item = (kind: Kind, index: number) => ({ ...(kind === "profile" ? profile : preset), id: `${kind}-${index}`, name: `${kind} ${index}` });
const clients: QueryClient[] = [];

function View() {
  const appearance = useAppearance();
  return <SettingsView engines={[]} appearance={appearance} destinationId="models-and-generation" onDestinationChange={() => undefined} />;
}
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  return render(<QueryClientProvider client={client}><View /></QueryClientProvider>);
}
beforeEach(() => {
  pages.profile.mockReset().mockResolvedValue([profile]);
  pages.preset.mockReset().mockResolvedValue([preset]);
  vi.mocked(api.profiles).mockResolvedValue([profile]);
  vi.mocked(api.presets).mockResolvedValue([preset]);
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); vi.restoreAllMocks(); });

it.each(kinds)("pages and searches %s choices using server matches", async (kind) => {
  pages[kind].mockImplementation(({ offset, search }: { offset: number; search: string }) => Promise.resolve(
    search ? [{ ...item(kind, 99), name: "Straße match" }]
      : offset === 0 ? Array.from({ length: 50 }, (_, index) => item(kind, index)) : [item(kind, 50)],
  ));
  show();
  fireEvent.click(await screen.findByRole("button", { name: `More ${label(kind)}` }));
  await screen.findByRole("button", { name: `Edit ${kind}: ${kind} 50` });
  expect(screen.getByRole("button", { name: `Edit ${kind}: ${kind} 0` })).toBeInTheDocument();
  fireEvent.change(screen.getByRole("searchbox", { name: `Search ${label(kind)}` }), { target: { value: "STRASSE" } });
  await screen.findByRole("button", { name: `Edit ${kind}: Straße match` });
  expect(pages[kind]).toHaveBeenLastCalledWith({ limit: 50, offset: 0, search: "STRASSE" });
  expect(screen.queryByRole("button", { name: `Edit ${kind}: ${kind} 0` })).not.toBeInTheDocument();
});

it.each(kinds)("retries an initial %s failure without showing an empty result", async (kind) => {
  pages[kind].mockRejectedValueOnce(new Error("Library unavailable"));
  const legacy = kind === "profile" ? api.profiles : api.presets;
  vi.mocked(legacy).mockRejectedValueOnce(new Error("Library unavailable"));
  show();
  await screen.findByText("Library unavailable");
  expect(screen.queryByText(`No ${label(kind)} found.`)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: `Retry ${label(kind)}` }));
  await screen.findByRole("button", { name: `Edit ${kind}: Neutral ${kind}` });
});

it.each(kinds)("retains loaded %s choices when the next page fails", async (kind) => {
  pages[kind].mockResolvedValueOnce(Array.from({ length: 50 }, (_, index) => item(kind, index)))
    .mockRejectedValueOnce(new Error("Next page unavailable")).mockResolvedValueOnce([item(kind, 50)]);
  show();
  fireEvent.click(await screen.findByRole("button", { name: `More ${label(kind)}` }));
  await screen.findByText("Next page unavailable");
  expect(screen.getByRole("button", { name: `Edit ${kind}: ${kind} 0` })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: `Retry ${label(kind)}` }));
  await screen.findByRole("button", { name: `Edit ${kind}: ${kind} 50` });
  expect(pages[kind]).toHaveBeenLastCalledWith({ limit: 50, offset: 50, search: "" });
});

it("refreshes paged profile defaults after changing the default", async () => {
  vi.mocked(api.updateProfile).mockImplementation(async () => {
    const updated = { ...profile, is_default: true };
    pages.profile.mockResolvedValue([updated]);
    return updated;
  });
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Set Neutral profile as default chat model" }));
  await screen.findByText("Neutral profile · default");
  expect(pages.profile).toHaveBeenCalledTimes(2);
});

it("refreshes paged presets after cloning an edited preset", async () => {
  vi.mocked(api.clonePreset).mockImplementation(async () => {
    const clone = { ...preset, id: "clone", name: "Cloned preset" };
    pages.preset.mockResolvedValue([preset, clone]);
    return clone;
  });
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Edit preset: Neutral preset" }));
  fireEvent.click(screen.getByRole("button", { name: "Clone" }));
  await screen.findByRole("button", { name: "Edit preset: Cloned preset" });
  expect(pages.preset).toHaveBeenCalledTimes(2);
});

it.each(kinds)("refreshes the paged %s library after importing a neutral bundle", async (kind) => {
  const imported = { ...item(kind, 70), name: "Imported choice" };
  vi.mocked(api.importProfile).mockImplementation(async () => {
    pages.profile.mockResolvedValue([imported]); return { ...profile, name: imported.name };
  });
  vi.mocked(api.importPreset).mockImplementation(async () => {
    pages.preset.mockResolvedValue([imported]); return { ...preset, name: imported.name };
  });
  const view = show();
  await screen.findByRole("button", { name: `Edit ${kind}: Neutral ${kind}` });
  const file = new File(["{}"], "neutral-library.json", { type: "application/json" });
  Object.defineProperty(file, "text", { value: async () => JSON.stringify({ format: `lm-atelier-${kind}` }) });
  const inputs = view.container.querySelectorAll<HTMLInputElement>('input[type="file"]');
  fireEvent.change(inputs[kind === "profile" ? 0 : 1], { target: { files: [file] } });
  await screen.findByRole("button", { name: `Edit ${kind}: Imported choice` });
  await waitFor(() => expect(pages[kind]).toHaveBeenCalledTimes(2));
});
