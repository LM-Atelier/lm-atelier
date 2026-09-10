import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Workflow, WorkflowRevision } from "./types";

vi.mock("./api", () => ({ api: {
  workflows: vi.fn(), workflowFamilies: vi.fn(), validateWorkflow: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));

function revision(version: number): WorkflowRevision {
  return { id: "revision-" + version, workflow_id: "landscape", version, engine: "comfyui",
    engine_version: null, api_graph_json: { version }, ui_graph_json: {}, input_schema_json: {},
    dependencies_json: {}, trusted: true, created_at: "2026-09-01T00:00:00Z" };
}
function workflow(): Workflow {
  return { id: "landscape", name: "Landscape study", description: "Neutral fixture",
    operation: "text_to_image", current_revision_id: "revision-2", revisions: [revision(1), revision(2)] };
}
const success = "Workflow and declared dependencies are valid for the active media engine.";
const clients: QueryClient[] = [];
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.workflows).mockResolvedValue([workflow()]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.validateWorkflow).mockResolvedValue({
    valid: true, errors: [], warnings: [], revision_id: "revision-2",
  });
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

async function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
  fireEvent.click(await screen.findByText("Landscape study"));
  return client;
}
function inspect(id: string) { fireEvent.change(screen.getByLabelText("Revision"), { target: { value: id } }); }

describe("validation belongs to the returned current revision", () => {
  it("shows a matching current result and hides it while inspecting an older revision", async () => {
    await open();
    fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    expect(await screen.findByText(success)).toBeInTheDocument();
    inspect("revision-1");
    expect(screen.queryByText(success)).not.toBeInTheDocument();
    inspect("revision-2");
    expect(screen.getByText(success)).toBeInTheDocument();
  });

  it("explains why historical validation is unavailable and sends no request", async () => {
    await open();
    inspect("revision-1");
    expect(screen.getByRole("button", { name: "Validate" })).toBeDisabled();
    expect(screen.getByText("Select the current revision to validate it.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    expect(api.validateWorkflow).not.toHaveBeenCalled();
    inspect("revision-2");
    expect(screen.getByRole("button", { name: "Validate" })).toBeEnabled();
    expect(screen.queryByText("Select the current revision to validate it.")).not.toBeInTheDocument();
  });

  it.each([true, false])("hides a mismatched returned revision when valid is %s", async (valid) => {
    let finish!: (value: Awaited<ReturnType<typeof api.validateWorkflow>>) => void;
    vi.mocked(api.validateWorkflow).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    await open();
    fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    await waitFor(() => expect(api.validateWorkflow).toHaveBeenCalledOnce());
    expect(await screen.findByRole("button", { name: "Validating..." })).toBeDisabled();
    await act(async () => { finish({
      valid, errors: ["Different revision has a missing node"], warnings: ["Different revision warning"],
      revision_id: "revision-3",
    }); });
    await screen.findByRole("button", { name: "Validate" });
    expect(screen.queryByText(success)).not.toBeInTheDocument();
    expect(screen.queryByText(/Different revision/)).not.toBeInTheDocument();
  });

  it("does not attach an in-flight result to a historical selection", async () => {
    let finish!: (value: Awaited<ReturnType<typeof api.validateWorkflow>>) => void;
    vi.mocked(api.validateWorkflow).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    await open();
    fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    await waitFor(() => expect(api.validateWorkflow).toHaveBeenCalledOnce());
    expect(await screen.findByRole("button", { name: "Validating..." })).toBeDisabled();
    inspect("revision-1");
    await act(async () => { finish({ valid: true, errors: [], warnings: [], revision_id: "revision-2" }); });
    await screen.findByRole("button", { name: "Validate" });
    expect(screen.queryByText(success)).not.toBeInTheDocument();
  });

  it("drops the old verdict when the same workflow receives a new current revision", async () => {
    const client = await open();
    fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    expect(await screen.findByText(success)).toBeInTheDocument();
    const updated = workflow();
    updated.revisions.push(revision(3)); updated.current_revision_id = "revision-3";
    await act(async () => { client.setQueryData(["workflows"], [updated]); });
    await screen.findByRole("option", { name: /^v3/ });
    expect(screen.queryByText(success)).not.toBeInTheDocument();
    inspect("revision-3");
    expect(screen.queryByText(success)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Validate" })).toBeEnabled();
  });
});
