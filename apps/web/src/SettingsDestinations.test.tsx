import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { SettingsView } from "./SettingsView";
import { SETTINGS_DESTINATIONS, settingsDestinationFor } from "./settingsDestinations";

vi.mock("./api", () => ({ api: {
  system: vi.fn().mockResolvedValue(null),
  about: vi.fn().mockResolvedValue(null),
  profiles: vi.fn().mockResolvedValue([]),
  presets: vi.fn().mockResolvedValue([]),
  workers: vi.fn().mockResolvedValue([]),
  runtimes: vi.fn().mockResolvedValue([]),
  backups: vi.fn().mockResolvedValue([]),
  credentialStatus: vi.fn().mockResolvedValue({ configured: false, vault_available: true }),
  workerSettings: vi.fn().mockResolvedValue({ worker_startup_seconds: 60 }),
} }));

const clients: QueryClient[] = [];

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
});

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><SettingsView engines={[]} /></QueryClientProvider>);
}

function rail(name: string) {
  return screen.getByRole("button", { name });
}

it("offers every destination and announces which one you are on", () => {
  show();

  for (const destination of SETTINGS_DESTINATIONS) {
    expect(rail(destination.label)).toBeTruthy();
  }
  // Pages, not tabs: the current one is the current PAGE, and exactly one is.
  const current = screen.getAllByRole("button").filter(
    (button) => button.getAttribute("aria-current") === "page",
  );
  expect(current).toHaveLength(1);
  expect(current[0].textContent).toBe(SETTINGS_DESTINATIONS[0].label);
});

it("shows one destination at a time", () => {
  show();

  // Models & generation is where Settings opens, so its content is present and
  // another destination's is not - the whole point of the shell.
  expect(screen.getByRole("heading", { name: "Generation presets" })).toBeTruthy();
  expect(screen.queryByRole("heading", { name: "Recovery backups" })).toBeNull();

  fireEvent.click(rail("Data & backups"));

  expect(screen.getByRole("heading", { name: "Recovery backups" })).toBeTruthy();
  expect(screen.queryByRole("heading", { name: "Generation presets" })).toBeNull();
});

it("moves focus to the destination you chose, and not before you choose one", () => {
  show();

  // Nothing is stolen on arrival: whatever opened Settings keeps focus.
  expect(document.activeElement).toBe(document.body);

  fireEvent.click(rail("Advanced"));

  const region = screen.getByRole("region", { name: "Advanced" });
  expect(document.activeElement).toBe(region);
});

it("the picker and the rail are the same choice", () => {
  show();

  const picker = screen.getByLabelText("Settings section");
  fireEvent.change(picker, { target: { value: "about-and-support" } });

  expect(screen.getByRole("heading", { name: "About & support" })).toBeTruthy();
  expect(rail("About & support").getAttribute("aria-current")).toBe("page");
  expect((picker as HTMLSelectElement).value).toBe("about-and-support");
});

it("an id that no longer exists lands somewhere usable", () => {
  // A destination can be renamed or retired between releases. Somebody
  // returning to a remembered one should not get a blank page.
  expect(settingsDestinationFor("a-destination-that-was-retired").id)
    .toBe(SETTINGS_DESTINATIONS[0].id);
  expect(settingsDestinationFor(undefined).id).toBe(SETTINGS_DESTINATIONS[0].id);
  expect(settingsDestinationFor("advanced").label).toBe("Advanced");
});
