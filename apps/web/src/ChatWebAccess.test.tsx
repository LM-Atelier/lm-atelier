import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api } from "./api";
import { ChatWebAccess } from "./ChatWebAccess";
import type { Chat, WebSearchConfiguration, WebSettings } from "./types";

vi.mock("./api", () => ({ api: { searchConfiguration: vi.fn(), updateChat: vi.fn() } }));
const clients: QueryClient[] = [];
const ready: WebSearchConfiguration = {
  installation_enabled: true, configured: true, provider: "CRW",
  provider_endpoint: "https://search.example.test", error_code: null,
};
const off: WebSettings = {
  allow_url_fetch: false, allow_search: false, allow_search_without_asking: false,
};
beforeEach(() => vi.resetAllMocks());
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });
async function mount(settings = off, configuration = ready) {
  vi.mocked(api.searchConfiguration).mockResolvedValue(configuration);
  const chat = { id: "chat-one", web_settings_json: settings } as Chat;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><ChatWebAccess chat={chat} /></QueryClientProvider>);
  fireEvent.click(screen.getByText("Web access"));
  await waitFor(() => expect(screen.queryByText("Checking web access…")).not.toBeInTheDocument());
  return client;
}

it("starts off and saves search permission independently of reading links", async () => {
  vi.mocked(api.updateChat).mockResolvedValue({ id: "chat-one" } as Chat);
  const client = await mount();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  expect(screen.getByLabelText("Read links I include in messages")).not.toBeChecked();
  expect(screen.getByLabelText("Allow searches without asking again")).toBeDisabled();
  expect(api.updateChat).not.toHaveBeenCalled();
  fireEvent.click(screen.getByLabelText("Allow web searches"));
  await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith("chat-one", {
    web_settings_json: { ...off, allow_search: true },
  }));
  await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ["chat"] }));
});

it("keeps the installation off switch authoritative", async () => {
  await mount(off, { ...ready, installation_enabled: false });
  expect(screen.getByText("Web access is turned off for this installation.")).toBeVisible();
  for (const label of ["Allow web searches", "Read links I include in messages", "Allow searches without asking again"]) {
    expect(screen.getByLabelText(label)).toBeDisabled();
    fireEvent.click(screen.getByLabelText(label));
  }
  expect(api.updateChat).not.toHaveBeenCalled();
});

it("allows link reading when the search service is not configured", async () => {
  await mount(off, { ...ready, configured: false, provider_endpoint: null, error_code: "search_not_configured" });
  expect(screen.getByLabelText("Allow web searches")).toBeDisabled();
  expect(screen.getByLabelText("Read links I include in messages")).toBeEnabled();
});

it("revokes automatic approval together with search without changing link permission", async () => {
  await mount({ allow_search: true, allow_search_without_asking: true, allow_url_fetch: true });
  fireEvent.click(screen.getByLabelText("Allow web searches"));
  await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith("chat-one", {
    web_settings_json: { ...off, allow_url_fetch: true },
  }));
});

it("allows explicit revocation even while installation web access is off", async () => {
  await mount({ ...off, allow_search: true }, { ...ready, installation_enabled: false });
  expect(screen.getByLabelText("Allow web searches")).toBeEnabled();
  fireEvent.click(screen.getByLabelText("Allow web searches"));
  await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith("chat-one", { web_settings_json: off }));
});

it("requires a separate choice for automatic searches", async () => {
  await mount({ ...off, allow_search: true });
  fireEvent.click(screen.getByLabelText("Allow searches without asking again"));
  await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith("chat-one", {
    web_settings_json: { ...off, allow_search: true, allow_search_without_asking: true },
  }));
});

it("reports failed saves without pretending the permission changed", async () => {
  vi.mocked(api.updateChat).mockRejectedValue(new Error("Could not save chat permissions."));
  await mount();
  fireEvent.click(screen.getByLabelText("Allow web searches"));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not save chat permissions.");
  expect(screen.getByLabelText("Allow web searches")).not.toBeChecked();
});

it("keeps the active permission focusable while ignoring duplicate saves", async () => {
  let finish!: (chat: Chat) => void;
  vi.mocked(api.updateChat).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  await mount();
  const checkbox = screen.getByLabelText("Allow web searches");
  checkbox.focus();
  fireEvent.click(checkbox);
  await waitFor(() => expect(api.updateChat).toHaveBeenCalledTimes(1));
  expect(checkbox).toBeEnabled();
  expect(checkbox).toHaveFocus();
  expect(screen.getByRole("group", { name: "Permissions for this chat" }))
    .toHaveAttribute("aria-disabled", "true");
  fireEvent.click(checkbox);
  expect(api.updateChat).toHaveBeenCalledTimes(1);
  finish({ id: "chat-one" } as Chat);
  await waitFor(() => expect(screen.getByRole("group", { name: "Permissions for this chat" }))
    .toHaveAttribute("aria-disabled", "false"));
});
