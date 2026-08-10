import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { api } from "./api";
import { CustomNodesPanel } from "./CustomNodesPanel";
import type { CustomNodeInstall } from "./types";

vi.mock("./api", () => ({
  api: {
    customNodes: vi.fn(),
    installCustomNode: vi.fn(),
    updateCustomNode: vi.fn(),
    trustCustomNode: vi.fn(),
    rollbackCustomNode: vi.fn(),
    removeCustomNode: vi.fn(),
  },
}));

const node = (overrides: Partial<CustomNodeInstall> = {}): CustomNodeInstall => ({
  id: "node-kjnodes",
  name: "comfyui-kjnodes",
  source_url: "https://github.com/example/comfyui-kjnodes.git",
  revision: "a".repeat(40),
  previous_revision: null,
  tree_hash: "b".repeat(40),
  trusted: false,
  active: true,
  security_json: { review_required: true },
  created_at: "2026-08-10T00:00:00Z",
  updated_at: "2026-08-10T00:00:00Z",
  ...overrides,
});

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <CustomNodesPanel />
    </QueryClientProvider>,
  );
}

describe("CustomNodesPanel reviewed node inventory", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("binds canonical reviewed node types to the exact trusted revision", async () => {
    vi.mocked(api.customNodes).mockResolvedValue([node()]);
    vi.mocked(api.trustCustomNode).mockResolvedValue(node({
      trusted: true,
      security_json: { node_types: ["GetNode", "SetNode"] },
    }));
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Trust revision" }));
    const inventory = screen.getByRole("textbox", { name: "Reviewed node types" });
    fireEvent.change(inventory, { target: { value: "SetNode\nGetNode\nSetNode" } });
    fireEvent.click(screen.getByRole("button", {
      name: "I reviewed this revision - trust it",
    }));

    await waitFor(() => expect(api.trustCustomNode).toHaveBeenCalledWith(
      "node-kjnodes",
      true,
      ["GetNode", "SetNode"],
    ));
  });

  it("shows the previously reviewed inventory when trust is renewed", async () => {
    vi.mocked(api.customNodes).mockResolvedValue([node({
      security_json: { node_types: ["GetNode", "SetNode"] },
    })]);
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Trust revision" }));

    expect(screen.getByRole("textbox", { name: "Reviewed node types" }))
      .toHaveValue("GetNode\nSetNode");
  });
});
