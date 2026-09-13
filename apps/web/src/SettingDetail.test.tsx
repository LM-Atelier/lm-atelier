/** Settings editors open at the level chosen in Models & generation, and a switch inside one stays there. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { SETTING_DETAIL_KEY, storedSettingDetail } from "./settingDetail";
import { SettingsDrawer } from "./SettingsDrawer";
import { SettingsView } from "./SettingsView";
import { useAppearance } from "./theme";
import type { EngineCapabilities, EngineRole, SettingField } from "./types";

vi.mock("./api", () => ({ api: {
  system: vi.fn().mockResolvedValue(null),
  about: vi.fn().mockResolvedValue(null),
  profiles: vi.fn(),
  presets: vi.fn(),
  workers: vi.fn().mockResolvedValue([]),
  runtimes: vi.fn().mockResolvedValue([]),
  backups: vi.fn().mockResolvedValue([]),
  credentialStatus: vi.fn().mockResolvedValue({ configured: false, vault_available: true }),
  workerSettings: vi.fn().mockResolvedValue({ worker_startup_seconds: 60 }),
  artifactStorage: vi.fn().mockResolvedValue(null),
  modelStorage: vi.fn().mockResolvedValue(null),
  workflowRevisionOutputGeometry: vi.fn(),
  resolveWorkflowRevisionOutputGeometry: vi.fn(),
} }));

function field(key: string, label: string, visibility: SettingField["visibility"]): SettingField {
  return {
    key, label, type: "integer", default: 1, minimum: 0, maximum: 100, step: 1, choices: [],
    scope: "request", visibility, restart_required: false, available: true, unavailable_reason: null, help: "",
  };
}

const ENGINES = [{
  engine: "mock",
  version: "1",
  roles: ["image"],
  operations: [],
  healthy: true,
  settings: [field("steps", "Steps", "basic"), field("guidance", "Guidance", "advanced"), field("seed", "Seed", "expert")],
}] as unknown as EngineCapabilities[];

function Settings() {
  const appearance = useAppearance();
  return <SettingsView engines={ENGINES} appearance={appearance} destinationId="models-and-generation" onDestinationChange={() => undefined} />;
}

function showSettings() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(<QueryClientProvider client={client}><Settings /></QueryClientProvider>);
}

function pressed(group: HTMLElement): string | undefined {
  return within(group).getAllByRole("button").find((button) => button.getAttribute("aria-pressed") === "true")?.textContent ?? undefined;
}

beforeEach(() => {
  localStorage.clear();
  vi.mocked(api.presets).mockResolvedValue([
    { id: "preset-1", name: "Everyday", role: "image", settings_json: {}, is_default: false },
  ]);
  vi.mocked(api.profiles).mockResolvedValue([
    {
      id: "profile-1", model_install_id: null, name: "Painter", use_case: "", role: "image", engine: "mock",
      load_settings_json: {}, request_settings_json: {}, is_default: false,
    },
  ]);
});

afterEach(() => {
  cleanup();
});

it("starts at Basic, and remembers the level chosen in Models & generation", async () => {
  showSettings();
  const group = await screen.findByRole("group", { name: "Editors open at" });
  expect(pressed(group)).toBe("Basic");

  fireEvent.click(within(group).getByRole("button", { name: "Expert" }));

  expect(pressed(group)).toBe("Expert");
  expect(localStorage.getItem(SETTING_DETAIL_KEY)).toBe("expert");
});

it("opens a preset editor at the saved level, and a switch inside it stays with that editor", async () => {
  localStorage.setItem(SETTING_DETAIL_KEY, "advanced");
  showSettings();

  fireEvent.click(await screen.findByRole("button", { name: "Edit preset: Everyday" }));
  let dialog = await screen.findByRole("dialog", { name: "Edit preset" });
  const detail = within(dialog).getByRole("group", { name: "Preset setting detail" });
  expect(pressed(detail)).toBe("advanced");
  expect(within(dialog).getByText("Guidance")).toBeTruthy();
  expect(within(dialog).queryByText("Seed")).toBeNull();

  fireEvent.click(within(detail).getByRole("button", { name: "expert" }));
  expect(within(dialog).getByText("Seed")).toBeTruthy();
  expect(storedSettingDetail()).toBe("advanced");

  fireEvent.click(within(dialog).getByRole("button", { name: "Close preset editor" }));
  fireEvent.click(screen.getByRole("button", { name: "Edit preset: Everyday" }));
  dialog = await screen.findByRole("dialog", { name: "Edit preset" });
  expect(pressed(within(dialog).getByRole("group", { name: "Preset setting detail" }))).toBe("advanced");
});

it("opens a profile editor at the saved level", async () => {
  localStorage.setItem(SETTING_DETAIL_KEY, "expert");
  showSettings();

  fireEvent.click(await screen.findByRole("button", { name: "Edit profile: Painter" }));
  const dialog = await screen.findByRole("dialog", { name: "Edit profile" });

  expect(pressed(within(dialog).getByRole("group", { name: "Profile setting detail" }))).toBe("expert");
  expect(within(dialog).getByText("Seed")).toBeTruthy();
});

it("opens the chat settings at the saved level each time, whatever was chosen inside last time", () => {
  localStorage.setItem(SETTING_DETAIL_KEY, "advanced");
  const role: EngineRole = "image";
  const drawer = (open: boolean) => (
    <QueryClientProvider client={new QueryClient()}>
      <SettingsDrawer open={open} onClose={() => undefined} mode="image" role={role} onRole={() => undefined}
        engines={ENGINES} values={{}} onValues={() => undefined} presets={[]} presetId={null}
        onPreset={() => undefined} imageEdit={false} imageEditPrompt="" />
    </QueryClientProvider>
  );
  const view = render(drawer(true));
  const detail = screen.getByRole("group", { name: "Settings detail level" });
  expect(pressed(detail)).toBe("advanced");

  fireEvent.click(within(detail).getByRole("button", { name: "basic" }));
  expect(pressed(detail)).toBe("basic");

  view.rerender(drawer(false));
  view.rerender(drawer(true));
  expect(pressed(screen.getByRole("group", { name: "Settings detail level" }))).toBe("advanced");
  expect(storedSettingDetail()).toBe("advanced");
});

it("treats a level it does not recognise as Basic", () => {
  localStorage.setItem(SETTING_DETAIL_KEY, "everything");

  expect(storedSettingDetail()).toBe("basic");
});
