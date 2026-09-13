import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { WorkflowDiscover } from "./WorkflowDiscover";
import type { CatalogModel, CatalogPage } from "./types";

vi.mock("./api", () => ({ api: { workflowCatalog: vi.fn() } }));
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

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(
    <QueryClientProvider client={client}>
      <WorkflowDiscover />
    </QueryClientProvider>,
  );
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
