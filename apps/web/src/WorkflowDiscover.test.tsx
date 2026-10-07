import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { WorkflowDiscover } from "./WorkflowDiscover";
import type { CatalogModel, CatalogPage, Workflow, WorkflowPackageAnalysis } from "./types";

vi.mock("./api", () => ({
  api: {
    workflowCatalog: vi.fn(),
    workflowCatalogGraph: vi.fn(),
    analyzeWorkflowPackage: vi.fn(),
    ensureWorkflowPackageDraft: vi.fn(),
    importWorkflowPackage: vi.fn(),
  },
}));
const { api } = await import("./api");
const workflowCatalog = api.workflowCatalog as ReturnType<typeof vi.fn>;

const clients: QueryClient[] = [];

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  vi.clearAllMocks();
});

function item(overrides: Partial<CatalogModel> = {}): CatalogModel {
  return {
    provider: "civitai",
    remote_id: "801",
    name: "A portrait workflow",
    author: "someone",
    tags: ["comfyui", "flux"],
    downloads: 1234,
    likes: 56,
    formats: [],
    ...overrides,
  } as CatalogModel;
}

function page(overrides: Partial<CatalogPage> = {}): CatalogPage {
  return { items: [item()], next_cursor: null, stale: false, ...overrides } as CatalogPage;
}

function show(onImported?: () => void) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(
    <QueryClientProvider client={client}>
      <WorkflowDiscover onImported={onImported} />
    </QueryClientProvider>,
  );
}

const EXPORT = { nodes: [{ id: 1, type: "CheckpointLoaderSimple" }], links: [] };

function readyAnalysis(): WorkflowPackageAnalysis {
  return {
    format_version: "0.4",
    frontend_version: null,
    node_count: 1,
    link_count: 0,
    subgraph_count: 0,
    operation_guess: "image",
    truncated: false,
    required_node_types: ["CheckpointLoaderSimple"],
    frontend_node_types: [],
    missing_node_types: [],
    missing_nodes: [],
    custom_packages: [],
    asset_references: [],
    issues: [],
    ready: true,
    runtime_nodes_available: true,
    dependencies_resolved: true,
    node_inventory_available: true,
    source_candidates: [],
  };
}

it("does not claim the library is empty before anyone has searched", async () => {
  // An empty result and a search not yet run are different states. Saying
  // "no workflows" on arrival would be a claim about the library rather than
  // about the search, and it is wrong.
  workflowCatalog.mockResolvedValue(page({ items: [] }));
  show();

  expect(await screen.findByText("Search to find workflows you can add.")).toBeTruthy();
  expect(screen.queryByText(/No workflows match/)).toBeNull();
});

it("says which search found nothing, once one has been run", async () => {
  workflowCatalog.mockResolvedValue(page({ items: [] }));
  show();

  fireEvent.change(screen.getByLabelText("Search workflows"), {
    target: { value: "kontext" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search" }));

  expect(await screen.findByText(/No workflows match .*kontext/)).toBeTruthy();
});

it("shows a workflow without the model fields it never has", async () => {
  workflowCatalog.mockResolvedValue(page());
  show();

  expect(await screen.findByRole("heading", { name: "A portrait workflow" })).toBeTruthy();
  expect(screen.getByText("by someone")).toBeTruthy();
  expect(screen.getByText("1,234")).toBeTruthy();
  // The shared catalog type carries model vocabulary a workflow never sets.
  // Rendering those blank would read as missing information rather than as
  // information that does not apply.
  expect(screen.queryByText(/quantization/i)).toBeNull();
  expect(screen.queryByText(/parameters/i)).toBeNull();
  expect(screen.queryByText(/runtime/i)).toBeNull();
});

it("pages by cursor, carrying the same query and sort", async () => {
  // Answer by argument rather than by call order: the component also queries on
  // mount, so a call-ordered mock would be answering the wrong request.
  workflowCatalog.mockImplementation((_query: string, _sort: string, cursor?: string | null) =>
    Promise.resolve(
      cursor === "next"
        ? page({ items: [item({ remote_id: "802", name: "Another" })] })
        : page({ next_cursor: "next" }),
    ),
  );
  show();

  fireEvent.change(screen.getByLabelText("Search workflows"), { target: { value: "portrait" } });
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await screen.findByRole("heading", { name: "A portrait workflow" });

  fireEvent.click(screen.getByRole("button", { name: "Show more" }));

  expect(await screen.findByRole("heading", { name: "Another" })).toBeTruthy();
  // The continuation must keep asking the same question; a cursor spent under a
  // different query would silently mix two searches into one list.
  expect(workflowCatalog).toHaveBeenLastCalledWith("portrait", "trending", "next");
});

it("says results are out of date rather than presenting them as current", async () => {
  // An unreachable provider must not make the page look broken, and must not
  // quietly imply the list is fresh either.
  workflowCatalog.mockResolvedValue(page({ stale: true }));
  show();

  expect(await screen.findByRole("status")).toHaveTextContent(/may be out of date/);
  expect(screen.getByRole("heading", { name: "A portrait workflow" })).toBeTruthy();
});

it("surfaces a failed search instead of showing an empty library", async () => {
  workflowCatalog.mockRejectedValue(new Error("CivitAI is temporarily unavailable."));
  show();

  await waitFor(() =>
    expect(screen.getByText("CivitAI is temporarily unavailable.")).toBeTruthy(),
  );
  expect(screen.queryByText("Search to find workflows you can add.")).toBeNull();
});

it("submits the typed query only when the search is run", async () => {
  workflowCatalog.mockResolvedValue(page({ items: [] }));
  show();
  await screen.findByText("Search to find workflows you can add.");

  fireEvent.change(screen.getByLabelText("Search workflows"), { target: { value: "half typed" } });

  // Typing must not fire a request per keystroke at a rate-limited provider.
  expect(workflowCatalog).toHaveBeenCalledTimes(1);
  expect(workflowCatalog).toHaveBeenLastCalledWith("", "trending", null);
});

it("reviews a found workflow the way an imported file is reviewed", async () => {
  // A published workflow is somebody else's ComfyUI export, so it gets the same
  // review a downloaded one does, under the name it was published with.
  workflowCatalog.mockResolvedValue(page());
  vi.mocked(api.workflowCatalogGraph).mockResolvedValue({ version_id: "801", ui_graph: EXPORT });
  vi.mocked(api.analyzeWorkflowPackage).mockResolvedValue(readyAnalysis());
  show(vi.fn());

  fireEvent.click(await screen.findByRole("button", { name: "Review and add" }));

  const dialog = await screen.findByRole("dialog", { name: "Review workflow package" });
  expect(api.workflowCatalogGraph).toHaveBeenCalledWith("801", "civitai");
  expect(api.analyzeWorkflowPackage).toHaveBeenCalledWith(EXPORT);
  expect(within(dialog).getByDisplayValue("A portrait workflow")).toBeTruthy();
  // Nothing has been added by looking.
  expect(api.ensureWorkflowPackageDraft).not.toHaveBeenCalled();
  expect(api.importWorkflowPackage).not.toHaveBeenCalled();
});

it("imports the fetched graph itself and says so once it is added", async () => {
  workflowCatalog.mockResolvedValue(page());
  vi.mocked(api.workflowCatalogGraph).mockResolvedValue({ version_id: "801", ui_graph: EXPORT });
  vi.mocked(api.analyzeWorkflowPackage).mockResolvedValue(readyAnalysis());
  vi.mocked(api.ensureWorkflowPackageDraft).mockResolvedValue({
    id: "draft-1",
    current_revision_id: "draft-revision-1",
  } as Workflow);
  vi.mocked(api.importWorkflowPackage).mockResolvedValue({ id: "wf-1" } as Workflow);
  const onImported = vi.fn();
  show(onImported);

  fireEvent.click(await screen.findByRole("button", { name: "Review and add" }));
  const dialog = await screen.findByRole("dialog", { name: "Review workflow package" });
  fireEvent.click(within(dialog).getByRole("button", { name: "Import workflow" }));

  await waitFor(() => expect(onImported).toHaveBeenCalledOnce());
  expect(vi.mocked(api.importWorkflowPackage).mock.calls[0][0]).toMatchObject({
    ui_graph: EXPORT,
    name: "A portrait workflow",
  });
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("says why a workflow could not be fetched, and opens no review", async () => {
  workflowCatalog.mockResolvedValue(page());
  vi.mocked(api.workflowCatalogGraph).mockRejectedValue(
    new Error("This version carries more than one workflow graph, so which to install is ambiguous"),
  );
  show(vi.fn());

  fireEvent.click(await screen.findByRole("button", { name: "Review and add" }));

  expect(await screen.findByRole("alert")).toHaveTextContent(/more than one workflow graph/);
  expect(api.analyzeWorkflowPackage).not.toHaveBeenCalled();
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("fetches one graph at a time rather than one per click", async () => {
  workflowCatalog.mockResolvedValue(
    page({ items: [item(), item({ remote_id: "802", name: "Another" })] }),
  );
  vi.mocked(api.workflowCatalogGraph).mockReturnValue(new Promise(() => undefined));
  show(vi.fn());

  const [first, second] = await screen.findAllByRole("button", { name: "Review and add" });
  fireEvent.click(first);

  expect(await screen.findByRole("button", { name: "Fetching…" })).toBeTruthy();
  expect(second).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(second);
  fireEvent.click(screen.getByRole("button", { name: "Fetching…" }));
  // A mutation reaches its request only after a turn of the event loop, so a
  // second fetch would not have shown up in the count yet without this.
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(api.workflowCatalogGraph).toHaveBeenCalledOnce();
});
