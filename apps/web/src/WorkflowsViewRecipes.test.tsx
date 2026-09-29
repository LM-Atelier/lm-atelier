import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { WorkflowsView } from "./WorkflowsView";

vi.mock("./api", () => ({ api: {
  workflowSummaries: vi.fn(async () => []), workflowFamilies: vi.fn(async () => []),
  workflowUseCasePresets: vi.fn(async () => []), projects: vi.fn(async () => []),
  workflowUseCaseDefault: vi.fn(async () => ({ preset_id: null })),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
afterEach(cleanup);

it("opens recipe management from the workflow library", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Manage recipes" }));
  expect(await screen.findByRole("dialog", { name: "Workflow recipes" })).toBeVisible();
  expect(screen.getByRole("button", { name: "New recipe" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Close recipe manager" }));
  expect(screen.queryByRole("dialog", { name: "Workflow recipes" })).toBeNull();
  client.clear();
});
