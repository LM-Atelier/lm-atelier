/** About & support names the licence and credits the typefaces, as somebody reads the page. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { SettingsView } from "./SettingsView";
import { useAppearance } from "./theme";

vi.mock("./api", () => ({ api: {
  system: vi.fn().mockResolvedValue(null),
  about: vi.fn().mockResolvedValue({
    max_media_outputs_per_plan: 8,
    version: "0.1.8",
    web_access_enabled: false,
    data_directory: "/lm-atelier/data",
    log_directory: "/lm-atelier/data/logs",
    artifact_directory: "/lm-atelier/data/artifacts",
    artifact_directory_requested: null,
  }),
  profiles: vi.fn().mockResolvedValue([]),
  presets: vi.fn().mockResolvedValue([]),
  workers: vi.fn().mockResolvedValue([]),
  runtimes: vi.fn().mockResolvedValue([]),
  backups: vi.fn().mockResolvedValue([]),
  credentialStatus: vi.fn().mockResolvedValue({ configured: false, vault_available: true }),
  workerSettings: vi.fn().mockResolvedValue({ worker_startup_seconds: 60 }),
} }));

function About() {
  const appearance = useAppearance();
  return (
    <SettingsView engines={[]} appearance={appearance} destinationId="about-and-support"
      onDestinationChange={() => undefined} focusRequest={0} />
  );
}

afterEach(() => {
  cleanup();
});

it("links the licence and credits each typeface with the licence it ships under", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><About /></QueryClientProvider>);

  const resources = await screen.findByRole("navigation", { name: "Support resources" });
  expect(within(resources).getByRole("link", { name: "License" }).getAttribute("href"))
    .toBe("https://github.com/LM-Atelier/lm-atelier/blob/v0.1.8/LICENSE");

  const credits = screen.getByText(/Typefaces:/);
  expect(credits.textContent).toBe(
    "Typefaces: Inter, Source Serif 4 and JetBrains Mono, each under the SIL Open Font License.",
  );
  expect(within(credits).getByRole("link", { name: "Source Serif 4" }).getAttribute("href"))
    .toBe("/fonts/OFL-SourceSerif4.txt");
});
