import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { ChatSearchConsent } from "./ChatSearchConsent";
import type { WebSearch } from "./types";

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: { editSearch: vi.fn(), decideSearch: vi.fn() },
}));

const clients: QueryClient[] = [];
const pending: WebSearch = {
  run_id: "run-one", assistant_message_id: "message-one", job_id: "job-one", revision: 1,
  state: "awaiting_approval", query: "Compare brass and aluminum", provider: "CRW",
  provider_endpoint: "https://search.example.test", dispatch_after: null,
  results: [], result_count: 0, truncated: false, error_code: null,
};

beforeEach(() => vi.resetAllMocks());
afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
});

function mount(search = pending, onChanged = vi.fn(), onUseSource = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  clients.push(client);
  const view = render(
    <QueryClientProvider client={client}>
      <ChatSearchConsent search={search} onChanged={onChanged} onUseSource={onUseSource} />
    </QueryClientProvider>,
  );
  return {
    onChanged, onUseSource,
    update: (next: WebSearch) => view.rerender(
      <QueryClientProvider client={client}>
        <ChatSearchConsent search={next} onChanged={onChanged} onUseSource={onUseSource} />
      </QueryClientProvider>,
    ),
  };
}

it("shows the exact destination and waits for an explicit approval", async () => {
  vi.mocked(api.decideSearch).mockResolvedValue({ ...pending, state: "approved" });
  const { onChanged } = mount();
  expect(screen.getByLabelText("Exact query")).toHaveValue(pending.query);
  expect(screen.getByText("Provider: CRW at https://search.example.test")).toBeVisible();
  expect(api.decideSearch).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() => expect(api.decideSearch).toHaveBeenCalledWith("job-one", 1, "approve"));
  await waitFor(() => expect(onChanged).toHaveBeenCalledOnce());
});

it("saves an exact edited query before approval can use the new revision", async () => {
  const edited = { ...pending, revision: 2, query: "  Compare copper and steel  " };
  vi.mocked(api.editSearch).mockResolvedValue(edited);
  vi.mocked(api.decideSearch).mockResolvedValue({ ...edited, state: "approved" });
  const view = mount();
  fireEvent.change(screen.getByLabelText("Exact query"), { target: { value: edited.query } });
  const search = screen.getByRole("button", { name: "Search" });
  expect(search).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(search);
  expect(api.decideSearch).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Save query" }));
  await waitFor(() => expect(api.editSearch).toHaveBeenCalledWith("job-one", 1, edited.query));
  await waitFor(() => expect(view.onChanged).toHaveBeenCalledOnce());
  view.update(edited);
  expect(screen.getByLabelText("Exact query")).toHaveValue(edited.query);
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() => expect(api.decideSearch).toHaveBeenCalledWith("job-one", 2, "approve"));
});

it.each(["scheduled", "approved"] as const)("can cancel a %s search before dispatch", async (state) => {
  vi.mocked(api.decideSearch).mockResolvedValue({ ...pending, state: "cancelled" });
  mount({ ...pending, state });
  expect(screen.getByText(pending.query)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Cancel search" }));
  await waitFor(() => expect(api.decideSearch).toHaveBeenCalledWith("job-one", 1, "cancel"));
});

it("declines without requiring an edit to be saved", async () => {
  vi.mocked(api.decideSearch).mockResolvedValue({ ...pending, state: "declined" });
  mount();
  fireEvent.change(screen.getByLabelText("Exact query"), { target: { value: "Different query" } });
  fireEvent.click(screen.getByRole("button", { name: "Continue without search" }));
  await waitFor(() => expect(api.decideSearch).toHaveBeenCalledWith("job-one", 1, "decline"));
  expect(api.editSearch).not.toHaveBeenCalled();
});

it("refreshes a stale command and keeps its fixed error visible", async () => {
  vi.mocked(api.decideSearch).mockRejectedValue(
    new ApiError(409, "The search changed", "The search changed", "search-consent-conflict"),
  );
  const { onChanged } = mount();
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("The search changed");
  await waitFor(() => expect(onChanged).toHaveBeenCalledOnce());
});

it("prevents repeat submissions while keeping the pending button focusable", async () => {
  let finish: ((value: WebSearch) => void) | undefined;
  vi.mocked(api.decideSearch).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  const { onChanged } = mount();
  const search = screen.getByRole("button", { name: "Search" });
  search.focus();
  fireEvent.click(search);
  await waitFor(() => expect(search).toHaveAttribute("aria-disabled", "true"));
  expect(search).toBeEnabled();
  expect(search).toHaveFocus();
  fireEvent.click(search);
  expect(api.decideSearch).toHaveBeenCalledOnce();
  finish?.({ ...pending, state: "approved" });
  await waitFor(() => expect(onChanged).toHaveBeenCalledOnce());
});

it("offers result URLs only through the user's explicit choice", () => {
  const url = "https://example.test/source";
  const { onUseSource } = mount({
    ...pending, state: "complete", job_id: null, revision: null, result_count: 1,
    results: [{ url, title: "Material comparison", snippet: "A neutral summary" }],
  });
  const link = screen.getByRole("link", { name: "Material comparison" });
  expect(link).toHaveAttribute("href", url);
  expect(link).toHaveAttribute("rel", "noopener noreferrer");
  expect(onUseSource).not.toHaveBeenCalled();
  expect(api.decideSearch).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Add source to message" }));
  expect(onUseSource).toHaveBeenCalledWith(url);
});

it("does not offer history records as approval or repeat-search commands", () => {
  mount({ ...pending, state: "uncertain", job_id: null, revision: null });
  expect(screen.getByText("The previous request will not be sent again automatically.")).toBeVisible();
  expect(screen.queryByRole("button", { name: "Search" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Cancel search" })).not.toBeInTheDocument();
  expect(api.decideSearch).not.toHaveBeenCalled();
});


it.each(["copper\nsteel", "copper\tsteel", "copper\u0085steel", "copper\u2028steel", "copper\u2029steel"])(
  "explains invalid query characters and permits declining: %j", async (value) => {
    vi.mocked(api.decideSearch).mockResolvedValue({ ...pending, state: "declined" });
    mount();
    fireEvent.change(screen.getByLabelText("Exact query"), { target: { value } });
    expect(screen.getByRole("alert")).toHaveTextContent("one line");
    expect(screen.getByLabelText("Exact query")).toHaveAttribute("aria-invalid", "true");
    fireEvent.click(screen.getByRole("button", { name: "Save query" }));
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(api.editSearch).not.toHaveBeenCalled();
    expect(api.decideSearch).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Continue without search" }));
    await waitFor(() => expect(api.decideSearch).toHaveBeenCalledWith("job-one", 1, "decline"));
  },
);

it.each([
  "\u0645\u06cc\u200c\u0631\u0648\u0645", "\u0915\u094d\u200d\u0937",
  "\u2764\ufe0f", "\u{1f469}\u200d\u{1f4bb}", "\u1820\u180e\u1820",
  "caf\u00e9", "\u6750\u6599", "\u{1f600}",
])("preserves valid Unicode spelling through edit and approval: %s", async (query) => {
  const edited = { ...pending, query, revision: 2 };
  vi.mocked(api.editSearch).mockResolvedValue(edited);
  vi.mocked(api.decideSearch).mockResolvedValue({ ...edited, state: "approved" });
  const view = mount();
  fireEvent.change(screen.getByLabelText("Exact query"), { target: { value: query } });
  expect(screen.getByLabelText("Exact query")).toHaveAttribute("aria-invalid", "false");
  fireEvent.click(screen.getByRole("button", { name: "Save query" }));
  await waitFor(() => expect(api.editSearch).toHaveBeenCalledWith("job-one", 1, query));
  view.update(edited);
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() => expect(api.decideSearch).toHaveBeenCalledWith("job-one", 2, "approve"));
});

it.each([
  "\u200b", "\u200c", "\u200d", "\ufe0f", "\ufff9", "\u0600",
  "\ue000", "\u{f0000}", "\ufffe", "\ufdd0", "\u2800", "\u202e",
])("discloses non-rendering characters before approval: %j", (character) => {
  mount({ ...pending, query: "copper" + character + "steel" });
  const preview = screen.getByRole("region", { name: "Query formatting" });
  expect(preview).toHaveTextContent("copper");
  expect(preview).toHaveTextContent("U+" + character.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0"));
  expect(preview).toHaveTextContent("steel");
  expect(api.decideSearch).not.toHaveBeenCalled();
});

it("bounds preview work for a query that cannot be sent", () => {
  mount();
  fireEvent.change(screen.getByLabelText("Exact query"), {
    target: { value: "\u200b".repeat(2001) },
  });
  expect(screen.getByLabelText("Exact query")).toHaveAttribute("aria-invalid", "true");
  expect(screen.queryByRole("region", { name: "Query formatting" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Save query" }));
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  expect(api.editSearch).not.toHaveBeenCalled();
  expect(api.decideSearch).not.toHaveBeenCalled();
});

it("keeps RTL query text together and lists formatting positions separately", () => {
  const query = "\u0645\u06cc\u200c\u0631\u0648\u0645";
  mount({ ...pending, query });
  const original = screen.getByLabelText("Query as entered");
  expect(original.textContent).toBe(query);
  expect(original).toHaveAttribute("dir", "auto");
  expect(original.children).toHaveLength(0);
  expect(screen.getByRole("region", { name: "Query formatting" })).toHaveTextContent(
    "Character 3: non-joiner U+200C",
  );
});

it("associates changing formatting information with the query field", () => {
  mount();
  const field = screen.getByLabelText("Exact query");
  const described = field.getAttribute("aria-describedby")!.split(/\s+/)
    .map((id) => document.getElementById(id));
  const live = described.find((element) => element?.getAttribute("aria-live") === "polite");
  expect(live).toBeTruthy();
  expect(live).toHaveTextContent("");
  fireEvent.change(field, { target: { value: "a\u200cb" } });
  expect(live).toHaveTextContent("Character 2: non-joiner U+200C");
});

it.each(["\u0915\u093f\u0924\u093e\u092c", "\u05e9\u05b8\u05c1\u05dc\u05d5\u05b9\u05dd", "\u0e19\u0e49\u0e33"])(
  "does not break ordinary combining script into formatting labels: %s", (query) => {
    mount({ ...pending, query });
    expect(screen.getByLabelText("Exact query")).toHaveValue(query);
    expect(screen.queryByRole("region", { name: "Query formatting" })).not.toBeInTheDocument();
  },
);
