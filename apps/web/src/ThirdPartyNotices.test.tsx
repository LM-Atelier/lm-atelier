/** Third-party notices in About & support, read when opened. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ThirdPartyNotices } from "./ThirdPartyNotices";

vi.mock("./api", () => ({ api: { thirdPartyNotices: vi.fn() } }));

const INVENTORY = [
  "# Third-party notices",
  "",
  "| Ecosystem | Package | Version | Declared license |",
  "| --- | --- | --- | --- |",
  "| npm | react | 19.1.0 | MIT |",
  "| pypi | fastapi | 0.116.1 | MIT |",
].join("\n");

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><ThirdPartyNotices /></QueryClientProvider>);
}

function open() {
  const summary = screen.getByText("Third-party notices", { selector: "summary" });
  const details = summary.closest("details")!;
  details.open = true;
  fireEvent(details, new Event("toggle"));
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("reads nothing until opened, then lists the release's third-party software and its license folder", async () => {
  vi.mocked(api.thirdPartyNotices).mockResolvedValue({
    text: INVENTORY,
    license_folder: "C:/Program Files/LM Atelier/_internal/third-party-licenses",
  });
  show();
  expect(api.thirdPartyNotices).not.toHaveBeenCalled();

  open();

  expect(await screen.findByRole("cell", { name: "fastapi" })).toBeInTheDocument();
  expect(screen.getByRole("cell", { name: "react" })).toBeInTheDocument();
  expect(screen.getByText("C:/Program Files/LM Atelier/_internal/third-party-licenses")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Copy license texts folder" })).toBeInTheDocument();
  expect(screen.queryByText(/Installed releases of LM Atelier include/)).toBeNull();
});

it("says where the notices are found when running from source", async () => {
  vi.mocked(api.thirdPartyNotices).mockResolvedValue({ text: null, license_folder: null });
  show();

  open();

  expect(await screen.findByText(/Installed releases of LM Atelier include the third-party notices/)).toBeInTheDocument();
  expect(screen.queryByRole("table")).toBeNull();
  expect(screen.queryByRole("button", { name: "Copy license texts folder" })).toBeNull();
});

it("says so when the notices cannot be read", async () => {
  vi.mocked(api.thirdPartyNotices).mockRejectedValue(new Error("unreachable"));
  show();

  open();

  expect(await screen.findByRole("alert")).toHaveTextContent("Third-party notices are unavailable right now.");
});
