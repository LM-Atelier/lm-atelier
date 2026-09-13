/** What Settings says about the search provider, and which part of it is wrong. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
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
