import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowFamilyPreferences } from "./WorkflowFamilyPreferences";
import { api } from "./api";
import type { WorkflowFamily } from "./types";

vi.mock("./api", () => ({ api: { updateWorkflowFamily: vi.fn(), setWorkflowFamilyPreference: vi.fn() } }));

function family(id: string): WorkflowFamily {
  return {
    id: `family-${id}`, name: `Family ${id}`, description: "", use_case: "Neutral scenes", tags: [],
    enabled: true, archived: false, compatibility: false, preferences: [],
    created_at: "2026-09-08T00:00:00Z", updated_at: "2026-09-08T00:00:00Z",
    variants: [{ id: `workflow-${id}`, variant_key: "image", name: `Workflow ${id}`,
      operation: "text_to_image", current_revision_id: `revision-${id}`, current_revision_version: 1,
      engine: "comfyui", capabilities: ["image"], trusted: true, readiness: "ready", readiness_reason: null }],
  };
}
const clients: QueryClient[] = [];
function renderFamily(one: WorkflowFamily) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  client.setQueryData(["workflow-families", "library", false], [family("a"), family("b")]);
  client.setQueryData(["profiles"], []);
  client.setQueryData(["workflows"], []);
  const tree = (value: WorkflowFamily) => <QueryClientProvider client={client}><WorkflowFamilyPreferences family={value} /></QueryClientProvider>;
  const view = render(tree(one));
  return { client, changeFamily: (value: WorkflowFamily) => view.rerender(tree(value)) };
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.updateWorkflowFamily).mockResolvedValue({ ...family("a"), name: "Outdoor family", use_case: "Outdoor scenes" });
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

describe("editing workflow family metadata", () => {
  it("preserves a newer use case when saving only the family name", async () => {
    let server = family("a");
    vi.mocked(api.updateWorkflowFamily).mockImplementation(async (_id, values) => {
      server = { ...server, ...values };
      return server;
    });
    const { client } = renderFamily(server);
    fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
    server = { ...server, use_case: "A newer description from another window" };
    fireEvent.change(screen.getByRole("textbox", { name: "Family name" }), { target: { value: "Renamed family" } });
    fireEvent.click(screen.getByRole("button", { name: "Save family details" }));
    await screen.findByText("Family details saved.");
    expect(server.use_case).toBe("A newer description from another window");
    expect(api.updateWorkflowFamily).toHaveBeenCalledWith("family-a", { name: "Renamed family" });
    expect(client.getQueryData<WorkflowFamily>(["workflow-family", "family-a"])?.use_case).toBe(server.use_case);
  });

  it("persists an explicit clear of the family use case", async () => {
    renderFamily(family("a"));
    fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Use case" }), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Save family details" }));
    await waitFor(() => expect(api.updateWorkflowFamily).toHaveBeenCalledWith("family-a", { name: "Family a", use_case: "" }));
  });

  it("does not carry a cancelled use-case edit into a later name-only save", async () => {
    renderFamily(family("a"));
    fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Use case" }), { target: { value: "Cancelled draft" } });
    fireEvent.click(screen.getByRole("button", { name: "Cancel family changes" }));
    fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
    expect(screen.getByRole("textbox", { name: "Use case" })).toHaveValue("Neutral scenes");
    fireEvent.change(screen.getByRole("textbox", { name: "Family name" }), { target: { value: "Renamed family" } });
    fireEvent.click(screen.getByRole("button", { name: "Save family details" }));
    await waitFor(() => expect(api.updateWorkflowFamily).toHaveBeenCalledWith("family-a", { name: "Renamed family" }));
  });

  it("saves trimmed family details and refreshes the family and linked-profile views", async () => {
    const { client } = renderFamily(family("a"));
    fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Family name" }), { target: { value: "  Outdoor family  " } });
    fireEvent.change(screen.getByRole("textbox", { name: "Use case" }), { target: { value: " Outdoor scenes " } });
    fireEvent.click(screen.getByRole("button", { name: "Save family details" }));
    await waitFor(() => expect(api.updateWorkflowFamily).toHaveBeenCalledWith("family-a", { name: "Outdoor family", use_case: "Outdoor scenes" }));
    await waitFor(() => expect(client.getQueryData<WorkflowFamily[]>(["workflow-families", "library", false])?.find((one) => one.id === "family-a"))
      .toMatchObject({ name: "Outdoor family", use_case: "Outdoor scenes" }));
    expect(client.getQueryState(["profiles"])?.isInvalidated).toBe(true);
    expect(client.getQueryState(["workflows"])?.isInvalidated).toBe(true);
    expect(client.getQueryData<WorkflowFamily[]>(["workflow-families", "library", false])?.find((one) => one.id === "family-b")).toEqual(family("b"));
  });

  it("rejects a blank name and cancels without writing", () => {
    renderFamily(family("a"));
    fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Family name" }), { target: { value: "   " } });
    expect(screen.getByRole("button", { name: "Save family details" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel family changes" }));
    expect(screen.queryByRole("textbox", { name: "Family name" })).not.toBeInTheDocument();
    expect(api.updateWorkflowFamily).not.toHaveBeenCalled();
  });

  it("keeps failed edits through a refetch and retries the same values", async () => {
    vi.mocked(api.updateWorkflowFamily).mockRejectedValueOnce(new Error("Family details could not be saved"))
      .mockResolvedValueOnce({ ...family("a"), name: "Retained edit" });
    const { changeFamily } = renderFamily(family("a"));
    fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Family name" }), { target: { value: "Retained edit" } });
    fireEvent.click(screen.getByRole("button", { name: "Save family details" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Family details could not be saved");
    changeFamily({ ...family("a"), name: "Refetched name" });
    expect(screen.getByRole("textbox", { name: "Family name" })).toHaveValue("Retained edit");
    fireEvent.click(screen.getByRole("button", { name: "Save family details" }));
    await waitFor(() => expect(api.updateWorkflowFamily).toHaveBeenNthCalledWith(2, "family-a", { name: "Retained edit" }));
    await waitFor(() => expect(screen.queryByRole("textbox", { name: "Family name" })).not.toBeInTheDocument());
  });

  it("does not retarget a pending save or discard another family's draft", async () => {
    let finish: ((value: WorkflowFamily) => void) | undefined;
    vi.mocked(api.updateWorkflowFamily).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    const { changeFamily } = renderFamily(family("a"));
    fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Family name" }), { target: { value: "First family edit" } });
    fireEvent.click(screen.getByRole("button", { name: "Save family details" }));
    await waitFor(() => expect(api.updateWorkflowFamily).toHaveBeenCalledWith("family-a", { name: "First family edit" }));
    changeFamily(family("b"));
    fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Family name" }), { target: { value: "Second family draft" } });
    await act(async () => finish?.({ ...family("a"), name: "First family edit" }));
    expect(screen.getByRole("textbox", { name: "Family name" })).toHaveValue("Second family draft");
    expect(api.updateWorkflowFamily).toHaveBeenCalledTimes(1);
  });
});


it("labels a derived family use case until its text is edited", () => {
  renderFamily({ ...family("a"), use_case_derived: true });
  expect(screen.getByText("Derived from model metadata")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
  expect(screen.getByText("Derived from model metadata")).toBeInTheDocument();
  fireEvent.change(screen.getByRole("textbox", { name: "Family name" }), { target: { value: "Renamed family" } });
  expect(screen.getByText("Derived from model metadata")).toBeInTheDocument();
  fireEvent.change(screen.getByRole("textbox", { name: "Use case" }), { target: { value: "My own use case" } });
  expect(screen.queryByText("Derived from model metadata")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Cancel family changes" }));
  expect(screen.getByText("Derived from model metadata")).toBeInTheDocument();
});

it("does not label an authored family use case as derived", () => {
  renderFamily({ ...family("a"), use_case_derived: false });
  expect(screen.queryByText("Derived from model metadata")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
  expect(screen.queryByText("Derived from model metadata")).not.toBeInTheDocument();
});
