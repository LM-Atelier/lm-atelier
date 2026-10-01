import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { SettingsView } from "./SettingsView";
import { useAppearance } from "./theme";
import type { GenerationPreset } from "./types";

vi.mock("./api", () => ({ api: {
  system: vi.fn().mockResolvedValue(null),
  about: vi.fn().mockResolvedValue(null),
  profilesPage: vi.fn().mockResolvedValue([]),
  presetsPage: vi.fn().mockResolvedValue([]),
  workers: vi.fn().mockResolvedValue([]),
  runtimes: vi.fn().mockResolvedValue([]),
  backups: vi.fn().mockResolvedValue([]),
  credentialStatus: vi.fn().mockResolvedValue({ configured: false, vault_available: true }),
  workerSettings: vi.fn().mockResolvedValue({ worker_startup_seconds: 60 }),
  createPreset: vi.fn(),
} }));

const preset: GenerationPreset = {
  id: "neutral-preset", name: "Neutral preset", role: "chat", settings_json: {}, is_default: false,
};
const clients: QueryClient[] = [];

function TestSettings() {
  const appearance = useAppearance();
  return <SettingsView engines={[]} appearance={appearance} destinationId="models-and-generation" onDestinationChange={() => undefined} />;
}

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><TestSettings /></QueryClientProvider>);
}

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
});

it("retains creation focus and ignores pending or empty submissions", async () => {
  let resolveCreation: (value: GenerationPreset) => void = () => undefined;
  vi.mocked(api.createPreset).mockImplementation(() => new Promise((resolve) => {
    resolveCreation = resolve;
  }));
  show();
  const name = screen.getByRole("textbox", { name: "New preset name" });
  const create = screen.getByRole("button", { name: "Create preset" });
  fireEvent.change(name, { target: { value: preset.name } });
  create.focus();
  expect(create).toHaveFocus();
  fireEvent.click(create);
  await waitFor(() => expect(api.createPreset).toHaveBeenCalledExactlyOnceWith("chat", preset.name));
  expect(create).not.toBeDisabled();
  expect(create).toHaveAttribute("aria-disabled", "true");
  expect(create).toHaveFocus();
  fireEvent.click(create);
  fireEvent.click(create);
  expect(api.createPreset).toHaveBeenCalledTimes(1);
  resolveCreation(preset);
  await waitFor(() => expect(name).toHaveValue(""));
  expect(create).toHaveFocus();
  fireEvent.click(create);
  fireEvent.click(create);
  expect(api.createPreset).toHaveBeenCalledTimes(1);
});

it("keeps the name and focused create control available after a failed request", async () => {
  let rejectCreation: (reason: Error) => void = () => undefined;
  vi.mocked(api.createPreset).mockImplementationOnce(() => new Promise((_resolve, reject) => {
    rejectCreation = reject;
  })).mockResolvedValueOnce(preset);
  show();
  const name = screen.getByRole("textbox", { name: "New preset name" });
  const create = screen.getByRole("button", { name: "Create preset" });
  fireEvent.change(name, { target: { value: preset.name } });
  create.focus();
  fireEvent.click(create);
  await waitFor(() => expect(api.createPreset).toHaveBeenCalledTimes(1));
  expect(create).not.toBeDisabled();
  rejectCreation(new Error("Could not create the preset"));
  await screen.findByText("Could not create the preset");
  expect(name).toHaveValue(preset.name);
  expect(create).toHaveFocus();
  expect(create).not.toHaveAttribute("aria-disabled", "true");
  fireEvent.click(create);
  await waitFor(() => expect(api.createPreset).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(name).toHaveValue(""));
  expect(create).toHaveFocus();
});
