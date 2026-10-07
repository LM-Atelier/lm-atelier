import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api } from "./api";
import { SetupWizard } from "./SetupWizard";
import type { SetupReadinessReport } from "./types";

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: { recipes: vi.fn(), system: vi.fn(), jobs: vi.fn() },
}));

const clients: QueryClient[] = [];

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.recipes).mockResolvedValue([]);
  vi.mocked(api.system).mockResolvedValue(null as never);
  vi.mocked(api.jobs).mockResolvedValue([]);
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
});

function report(nextAction: string, message: string): SetupReadinessReport {
  return {
    version: 2,
    state: "action_required",
    roles: [{
      role: "chat",
      state: "action_required",
      verification_level: "generation_probe",
      engine: "llama.cpp",
      job_id: null,
      verification_id: null,
      install_id: "install_1",
      profile_id: null,
      workflow_revision_id: null,
      next_action: nextAction,
      checks: [{ code: "profile_missing", status: "fail", message, action: nextAction }],
    }],
  };
}

function show(nextAction: string, message: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(
    <QueryClientProvider client={client}>
      <SetupWizard
        report={report(nextAction, message)}
        onClose={() => undefined}
        onOpenModels={() => undefined}
        onOpenWorkflows={() => undefined}
      />
    </QueryClientProvider>,
  );
}

it("names the profile step instead of offering to choose a model again", () => {
  // The checklist says a profile is missing for a model that is already
  // installed. Labelling that button "Choose chat model" asks for the one thing
  // the person has already done, and hides the one thing they have not.
  show("create_profile", "No usable profile is bound to this model.");

  expect(screen.getByRole("button", { name: "Create profile" })).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Choose chat model" })).toBeNull();
});

it("still offers to choose a model when that is what is missing", () => {
  // The control: the fallback label is right for the action it was written for,
  // and this is what proves the new branch did not take its place.
  show("select_model", "No model is installed for this role.");

  expect(screen.getByRole("button", { name: "Choose chat model" })).toBeTruthy();
});
