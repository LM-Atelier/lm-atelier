import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import { CLOCK_KEY } from "./clockPreference";
import { CHAT_WIDTH_KEY, MODE_KEY, MOTION_KEY, ROOM_KEY } from "./theme";

vi.mock("./api", () => ({
  api: {
    searchConfiguration: vi.fn(),
    setupReadiness: vi.fn(),
    projects: vi.fn(),
    chats: vi.fn(),
    chat: vi.fn(),
    workPlans: vi.fn(),
    engines: vi.fn(),
    profiles: vi.fn(),
    presets: vi.fn(),
    workflows: vi.fn(),
    workflowFamilies: vi.fn(),
    chatWorkflowSelections: vi.fn(),
    projectWorkflowSelections: vi.fn(),
    about: vi.fn(),
    jobs: vi.fn(),
    system: vi.fn(),
    workers: vi.fn(),
    runtimes: vi.fn(),
    backups: vi.fn(),
  },
  connectEvents: vi.fn().mockResolvedValue(() => undefined),
}));

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  window.history.replaceState(null, "", "/");
  delete document.documentElement.dataset.mode;
  delete document.documentElement.dataset.room;
  vi.mocked(api.setupReadiness).mockResolvedValue({ version: 2, state: "ready", roles: [] });
  vi.mocked(api.system).mockResolvedValue(null as never);
  vi.mocked(api.about).mockResolvedValue({
    max_media_outputs_per_plan: 8,
    version: "0.1.8",
    web_access_enabled: false,
    data_directory: "/lm-atelier/data",
    log_directory: "/lm-atelier/data/logs",
    artifact_directory: "/lm-atelier/data/artifacts",
    artifact_directory_requested: null,
  });
  for (const list of [
    api.projects, api.chats, api.workPlans, api.engines, api.profiles, api.presets,
    api.workflows, api.workflowFamilies, api.chatWorkflowSelections,
    api.projectWorkflowSelections, api.jobs, api.workers, api.runtimes, api.backups,
  ]) {
    vi.mocked(list).mockResolvedValue([]);
  }
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
}

async function openSettings() {
  fireEvent.click(await screen.findByRole("button", { name: "Settings" }));
  // Settings opens on General; Appearance is one destination along.
  fireEvent.click(await screen.findByRole("button", { name: "Appearance" }));
  return screen.findByRole("region", { name: "Appearance" });
}

it("changes the light and the theme from Settings, with nothing floating over the work", async () => {
  localStorage.setItem(MODE_KEY, "dark");
  localStorage.setItem(ROOM_KEY, "north-light");
  renderApp();

  await screen.findByRole("button", { name: "Settings" });
  expect(screen.queryByRole("group", { name: "Appearance" })).toBeNull();
  expect(screen.queryByRole("button", { name: /^Switch to (light|dark)$/ })).toBeNull();

  await openSettings();
  fireEvent.click(screen.getByRole("button", { name: "Light" }));

  expect(document.documentElement.dataset.mode).toBe("light");
  expect(localStorage.getItem(MODE_KEY)).toBe("light");
  expect(screen.getByRole("button", { name: "Light" }).getAttribute("aria-pressed")).toBe("true");
  expect(screen.getByRole("button", { name: "Dark" }).getAttribute("aria-pressed")).toBe("false");

  fireEvent.change(screen.getByRole("combobox", { name: "Theme" }), { target: { value: "blue-hour" } });

  expect(document.documentElement.dataset.room).toBe("blue-hour");
  expect(localStorage.getItem(ROOM_KEY)).toBe("blue-hour");
  // The room did not change the light, and the light did not change the room.
  expect(document.documentElement.dataset.mode).toBe("light");
});

it("opens on the choice somebody already made, rather than resetting it", async () => {
  // The control moved; the keys it remembers under did not. Anyone who chose a
  // light and a theme before the move still has them after it.
  localStorage.setItem(MODE_KEY, "light");
  localStorage.setItem(ROOM_KEY, "blue-hour");
  renderApp();

  await screen.findByRole("button", { name: "Settings" });
  expect(document.documentElement.dataset.mode).toBe("light");
  expect(document.documentElement.dataset.room).toBe("blue-hour");

  await openSettings();

  expect(screen.getByRole("button", { name: "Light" }).getAttribute("aria-pressed")).toBe("true");
  expect((screen.getByRole("combobox", { name: "Theme" }) as HTMLSelectElement).value).toBe("blue-hour");
});

it("offers System, which follows the computer's light setting", async () => {
  vi.stubGlobal("matchMedia", () => ({
    media: "(prefers-color-scheme: light)",
    matches: true,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
  localStorage.setItem(MODE_KEY, "dark");
  try {
    renderApp();
    await openSettings();

    fireEvent.click(screen.getByRole("button", { name: "System" }));

    expect(screen.getByRole("button", { name: "System" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "Dark" }).getAttribute("aria-pressed")).toBe("false");
    expect(localStorage.getItem(MODE_KEY)).toBe("system");
    expect(document.documentElement.dataset.mode).toBe("light");
  } finally {
    vi.unstubAllGlobals();
  }
});

it("writes times on the clock chosen in Settings, and shows how they will look", async () => {
  renderApp();
  await openSettings();
  const clock = screen.getByRole("group", { name: "Clock" });
  expect(within(clock).getByRole("button", { name: "Automatic" }).getAttribute("aria-pressed")).toBe("true");

  fireEvent.click(within(clock).getByRole("button", { name: "24-hour" }));

  expect(localStorage.getItem(CLOCK_KEY)).toBe("24");
  expect(within(clock).getByRole("button", { name: "24-hour" }).getAttribute("aria-pressed")).toBe("true");
  expect(within(clock).getByRole("button", { name: "Automatic" }).getAttribute("aria-pressed")).toBe("false");
  expect(screen.getByText(/Times look like 15:45\./)).toBeTruthy();
});

it("widens the chat from Settings and opens on that width next time", async () => {
  renderApp();
  await openSettings();
  const width = screen.getByRole("group", { name: "Chat width" });
  expect(within(width).getByRole("button", { name: "Standard" }).getAttribute("aria-pressed")).toBe("true");

  fireEvent.click(within(width).getByRole("button", { name: "Wide" }));

  expect(document.documentElement.dataset.chatWidth).toBe("wide");
  expect(localStorage.getItem(CHAT_WIDTH_KEY)).toBe("wide");
  expect(within(width).getByRole("button", { name: "Wide" }).getAttribute("aria-pressed")).toBe("true");
  cleanup();

  renderApp();
  await screen.findByRole("button", { name: "Settings" });
  expect(document.documentElement.dataset.chatWidth).toBe("wide");
});

it("reduces motion from Settings", async () => {
  renderApp();
  await openSettings();
  const motion = screen.getByRole("group", { name: "Motion" });
  expect(within(motion).getByRole("button", { name: "Automatic" }).getAttribute("aria-pressed")).toBe("true");

  fireEvent.click(within(motion).getByRole("button", { name: "Reduce" }));

  expect(document.documentElement.dataset.motion).toBe("reduced");
  expect(localStorage.getItem(MOTION_KEY)).toBe("reduce");
  expect(within(motion).getByRole("button", { name: "Reduce" }).getAttribute("aria-pressed")).toBe("true");
});
