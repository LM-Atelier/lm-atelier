/** Deleting a model should say what it breaks, not only what it removes. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { WorkflowConsumers } from "./WorkflowConsumers";
import type { WorkflowResourceConsumer } from "./types";

vi.mock("./api", () => ({ api: { workflowResourceConsumers: vi.fn() } }));

function consumer(overrides: Partial<WorkflowResourceConsumer> = {}): WorkflowResourceConsumer {
  return {
    workflow_id: "wf-1",
    workflow_name: "Portrait finish",
    workflow_family_id: "family-1",
    workflow_family_name: "Portraits",
    revision_ids: ["rev-1"],
    binding_count: 1,
    current_revision: true,
    ...overrides,
  };
}

function renderConsumers() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <WorkflowConsumers kind="model_install" resourceId="install-1" />
    </QueryClientProvider>,
  );
}

describe("WorkflowConsumers", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("names the workflows that need it to run", async () => {
    vi.mocked(api.workflowResourceConsumers).mockResolvedValue({
      resource_kind: "model_install",
      resource_id: "install-1",
      resource_name: "Some checkpoint",
      consumers: [consumer(), consumer({ workflow_id: "wf-2", workflow_name: "Landscape" })],
    });
    renderConsumers();

    expect(await screen.findByText("2 workflows needs this to run")).toBeInTheDocument();
    expect(screen.getByText("Portrait finish")).toBeInTheDocument();
    expect(screen.getByText("Landscape")).toBeInTheDocument();
  });

  it("separates what breaks next run from what merely mentions it", async () => {
    vi.mocked(api.workflowResourceConsumers).mockResolvedValue({
      resource_kind: "model_install",
      resource_id: "install-1",
      resource_name: "Some checkpoint",
      consumers: [
        consumer(),
        consumer({ workflow_id: "wf-2", workflow_name: "Old draft", current_revision: false }),
      ],
    });
    renderConsumers();

    // A workflow whose current revision needs this breaks next run; one where
    // only an older revision does keeps working, and conflating the two turns
    // a warning into an alarm.
    expect(await screen.findByText("1 workflow needs this to run")).toBeInTheDocument();
    expect(screen.getByText(/1 older revision also references it/)).toBeInTheDocument();
    expect(screen.queryByText("Old draft")).toBeNull();
  });

  it("says nothing at all when nothing uses it", async () => {
    vi.mocked(api.workflowResourceConsumers).mockResolvedValue({
      resource_kind: "model_install",
      resource_id: "install-1",
      resource_name: "Unused",
      consumers: [],
    });
    const { container } = renderConsumers();

    // An empty panel under every delete would train people to skip reading
    // the ones that matter.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(container.querySelector(".workflow-consumers")).toBeNull();
  });
});
