import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { SettingsView } from "./SettingsView";
import { useAppearance } from "./theme";
import type { GenerationPreset, ModelProfile } from "./types";

vi.mock("./api", () => ({ api: {
  system: vi.fn().mockResolvedValue(null),
  about: vi.fn().mockResolvedValue(null),
  profilesPage: vi.fn(),
  presetsPage: vi.fn(),
  workers: vi.fn().mockResolvedValue([]),
  runtimes: vi.fn().mockResolvedValue([]),
  backups: vi.fn().mockResolvedValue([]),
  credentialStatus: vi.fn().mockResolvedValue({ configured: false, vault_available: true }),
  workerSettings: vi.fn().mockResolvedValue({ worker_startup_seconds: 60 }),
  updateProfile: vi.fn(),
  updatePreset: vi.fn(),
} }));

const profile: ModelProfile = {
  id: "neutral-profile", model_install_id: null, name: "Neutral profile", use_case: "",
  role: "chat", engine: "mock", load_settings_json: {}, request_settings_json: {}, is_default: false,
};
const preset: GenerationPreset = {
  id: "neutral-preset", name: "Neutral preset", role: "chat", settings_json: {}, is_default: false,
};
const clients: QueryClient[] = [];
const kinds = ["profile", "preset"] as const;

function TestSettings() {
  const appearance = useAppearance();
  return <SettingsView engines={[]} appearance={appearance} destinationId="models-and-generation" onDestinationChange={() => undefined} />;
}

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><TestSettings /></QueryClientProvider>);
}

function holdSave() {
  let resolveSave: () => void = () => undefined;
  let rejectSave: (error: Error) => void = () => undefined;
  const pending = new Promise<void>((resolve, reject) => { resolveSave = resolve; rejectSave = reject; });
  vi.mocked(api.updateProfile).mockImplementation(() => pending.then(() => profile));
  vi.mocked(api.updatePreset).mockImplementation(() => pending.then(() => preset));
  return { resolveSave, rejectSave };
}

beforeEach(() => {
  vi.mocked(api.profilesPage).mockResolvedValue([profile]);
  vi.mocked(api.presetsPage).mockResolvedValue([preset]);
  vi.mocked(api.updateProfile).mockReset();
  vi.mocked(api.updatePreset).mockReset();
});
afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
});

it.each(kinds)("keeps the %s Save focused and refuses empty or repeated pending saves", async (kind) => {
  const { resolveSave } = holdSave();
  show();
  const update = kind === "profile" ? api.updateProfile : api.updatePreset;
  const launch = await screen.findByRole("button", { name: `Edit ${kind}: Neutral ${kind}` });
  launch.focus();
  fireEvent.click(launch);
  const name = screen.getByLabelText(kind === "profile" ? "Profile name" : "Preset name");
  const save = screen.getByRole("button", { name: `Save ${kind}` });
  fireEvent.change(name, { target: { value: " " } });
  await act(async () => { fireEvent.click(save); });
  expect(update).not.toHaveBeenCalled();
  fireEvent.change(name, { target: { value: "Renamed" } });
  save.focus();
  fireEvent.click(save);
  await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
  expect(save).not.toBeDisabled();
  expect(save).toHaveAttribute("aria-disabled", "true");
  expect(save).toHaveFocus();
  await act(async () => { fireEvent.click(save); fireEvent.click(save); });
  expect(update).toHaveBeenCalledTimes(1);
  resolveSave();
  await waitFor(() => expect(launch).toHaveFocus());
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it.each(kinds)("keeps the %s Save available for a focused retry after failure", async (kind) => {
  const { rejectSave } = holdSave();
  show();
  const update = kind === "profile" ? api.updateProfile : api.updatePreset;
  const launch = await screen.findByRole("button", { name: `Edit ${kind}: Neutral ${kind}` });
  launch.focus();
  fireEvent.click(launch);
  const save = screen.getByRole("button", { name: `Save ${kind}` });
  save.focus();
  fireEvent.click(save);
  await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
  expect(save).not.toBeDisabled();
  rejectSave(new Error("Could not save changes"));
  await screen.findByText("Could not save changes");
  expect(save).toHaveFocus();
  expect(save).toHaveAttribute("aria-disabled", "false");
  vi.mocked(api.updateProfile).mockResolvedValue(profile);
  vi.mocked(api.updatePreset).mockResolvedValue(preset);
  fireEvent.click(save);
  await waitFor(() => expect(update).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(launch).toHaveFocus());
});
