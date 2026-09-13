/** What Settings says about the search provider, and which part of it is wrong. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import type { WebSearchConfiguration } from "./types";
import { WebSearchSettings } from "./WebSearchSettings";

vi.mock("./api", () => ({
  api: {
    searchConfiguration: vi.fn(),
    credentialStatus: vi.fn().mockResolvedValue({ configured: false, vault_available: true }),
    setCredentialToken: vi.fn(),
    deleteCredentialToken: vi.fn(),
  },
}));

function show(configuration: WebSearchConfiguration) {
  vi.mocked(api.searchConfiguration).mockResolvedValue(configuration);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><WebSearchSettings /></QueryClientProvider>);
}

const REFUSED: WebSearchConfiguration = {
  installation_enabled: true,
  configured: false,
  provider: "CRW",
  provider_endpoint: null,
  error_code: null,
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("blames the token, not the address, when only the token is malformed", async () => {
  show({ ...REFUSED, error_code: "search_credentials_invalid" });

  expect(await screen.findByText(/The search token is not valid, so searches cannot be sent\./)).toBeInTheDocument();
  expect(screen.queryByText("The configured search provider address is invalid.")).toBeNull();
});

it("says the address is invalid when the address is", async () => {
  show({ ...REFUSED, error_code: "search_provider_invalid" });

  expect(await screen.findByText("The configured search provider address is invalid.")).toBeInTheDocument();
  expect(screen.queryByText(/The search token is not valid/)).toBeNull();
});

it("names the provider when it is configured, and says when none is", async () => {
  show({ ...REFUSED, configured: true, provider_endpoint: "https://search.example.test" });
  expect(await screen.findByText("Search provider: CRW at https://search.example.test")).toBeInTheDocument();
  cleanup();

  show({ ...REFUSED, error_code: "search_not_configured" });
  expect(await screen.findByText("No search provider is configured.")).toBeInTheDocument();
});

it("says whether this installation allows web access at all", async () => {
  show({ ...REFUSED, installation_enabled: false });
  expect(await screen.findByText("Web access is turned off for this installation.")).toBeInTheDocument();
  expect(screen.queryByText("Web access is available for this installation.")).toBeNull();
  cleanup();

  show(REFUSED);
  expect(await screen.findByText("Web access is available for this installation.")).toBeInTheDocument();
});

it("says it is checking, and alerts when the search status cannot be read", async () => {
  let fail: (error: Error) => void = () => undefined;
  vi.mocked(api.searchConfiguration).mockReturnValue(new Promise((_, reject) => { fail = reject; }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><WebSearchSettings /></QueryClientProvider>);

  expect(screen.getByRole("status")).toHaveTextContent("Checking search configuration…");
  fail(new Error("The search status could not be read."));

  expect(await screen.findByRole("alert")).toHaveTextContent("The search status could not be read.");
  expect(screen.queryByRole("status")).toBeNull();
});

it("asks again after the token changes, so a replaced token clears the warning", async () => {
  vi.mocked(api.searchConfiguration)
    .mockResolvedValueOnce({ ...REFUSED, error_code: "search_credentials_invalid" })
    .mockResolvedValue({ ...REFUSED, configured: true, provider_endpoint: "https://search.example.test" });
  vi.mocked(api.setCredentialToken).mockResolvedValue({
    provider: "crw", configured: true, source: "credential_vault", vault_available: true,
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><WebSearchSettings /></QueryClientProvider>);
  expect(await screen.findByText(/The search token is not valid/)).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText("CRW access token"), { target: { value: "replacement-token" } });
  fireEvent.click(screen.getByRole("button", { name: "Save CRW token" }));

  expect(await screen.findByText("Search provider: CRW at https://search.example.test")).toBeInTheDocument();
  expect(api.setCredentialToken).toHaveBeenCalledWith("crw", "replacement-token");
  expect(screen.queryByText(/The search token is not valid/)).toBeNull();

  const asked = vi.mocked(api.searchConfiguration).mock.calls.length;
  vi.mocked(api.deleteCredentialToken).mockResolvedValue({
    provider: "crw", configured: false, source: "none", vault_available: true,
  });
  fireEvent.click(await screen.findByRole("button", { name: "Remove CRW token" }));
  await waitFor(() => expect(vi.mocked(api.searchConfiguration).mock.calls.length).toBeGreaterThan(asked));
});
