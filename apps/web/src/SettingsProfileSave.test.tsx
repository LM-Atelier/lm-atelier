import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { SettingsView } from "./SettingsView";
import type { ModelProfile } from "./types";

vi.mock("./api", () => ({ api: {
  searchConfiguration: vi.fn().mockResolvedValue({
    installation_enabled: false, configured: false, provider: "CRW",
    provider_endpoint: null, error_code: "search_not_configured",
  }),
  system: vi.fn().mockResolvedValue(null),
  about: vi.fn().mockResolvedValue(null),
  profiles: vi.fn(),
  presets: vi.fn().mockResolvedValue([]),
  workers: vi.fn().mockResolvedValue([]),
  runtimes: vi.fn().mockResolvedValue([]),
  backups: vi.fn().mockResolvedValue([]),
  credentialStatus: vi.fn().mockResolvedValue({ configured: false, vault_available: true }),
  workerSettings: vi.fn().mockResolvedValue({ worker_startup_seconds: 60 }),
  updateProfile: vi.fn(),
} }));

const profile: ModelProfile = {
  id: "neutral-profile", model_install_id: null, name: "Neutral profile",
  use_case: "coding; technical answers",
  role: "chat", engine: "mock", load_settings_json: {}, request_settings_json: {},
  is_default: false,
};

function show(value = profile) {
  vi.mocked(api.profiles).mockResolvedValue([value]);
  vi.mocked(api.updateProfile).mockResolvedValue(value);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(<QueryClientProvider client={client}><SettingsView engines={[]} /></QueryClientProvider>);
  return client;
}

beforeEach(() => vi.clearAllMocks());
afterEach(cleanup);

it("preserves a newer server description on a name-only save and refreshes family views", async () => {
  const client = show();
  let serverDescription = profile.use_case;
  vi.mocked(api.updateProfile).mockImplementation(async (_id, payload) => {
    if (payload.use_case !== undefined) serverDescription = payload.use_case;
    return { ...profile, name: payload.name ?? profile.name, use_case: serverDescription };
  });
  const invalidate = vi.spyOn(client, "invalidateQueries");
  fireEvent.click(await screen.findByRole("button", { name: "Edit profile: Neutral profile" }));
  serverDescription = "A description saved from another window";
  fireEvent.change(screen.getByLabelText("Profile name"), { target: { value: "Renamed profile" } });
  fireEvent.click(screen.getByRole("button", { name: "Save profile" }));
  await waitFor(() => expect(api.updateProfile).toHaveBeenCalled());
  const [id, payload] = vi.mocked(api.updateProfile).mock.calls[0];
  expect(id).toBe(profile.id);
  expect(payload.name).toBe("Renamed profile");
  expect(serverDescription).toBe("A description saved from another window");
  expect(payload).not.toHaveProperty("use_case");
  await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ["workflow-families"] }));
});

it("sends a deliberate clear as user text and retains it after a refused save", async () => {
  show();
  vi.mocked(api.updateProfile).mockRejectedValue(new Error("Could not save the profile"));
  fireEvent.click(await screen.findByRole("button", { name: "Edit profile: Neutral profile" }));
  fireEvent.change(screen.getByLabelText("Best used for"), { target: { value: "" } });
  fireEvent.click(screen.getByRole("button", { name: "Save profile" }));
  await screen.findByText("Could not save the profile");
  expect(api.updateProfile).toHaveBeenCalledWith(profile.id, expect.objectContaining({ use_case: "" }));
  expect(screen.getByLabelText("Best used for")).toHaveValue("");
});

it("sends an explicitly edited description even when changed back to its original text", async () => {
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Edit profile: Neutral profile" }));
  fireEvent.change(screen.getByLabelText("Best used for"), { target: { value: "Temporary revision" } });
  fireEvent.change(screen.getByLabelText("Best used for"), { target: { value: profile.use_case } });
  fireEvent.click(screen.getByRole("button", { name: "Save profile" }));
  await waitFor(() => expect(api.updateProfile).toHaveBeenCalledWith(
    profile.id, expect.objectContaining({ use_case: profile.use_case }),
  ));
});
