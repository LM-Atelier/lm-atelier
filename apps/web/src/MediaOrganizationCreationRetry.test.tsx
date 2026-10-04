import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { MediaOrganizationManager } from "./MediaOrganizationManager";
import { useMediaOrganization } from "./useMediaOrganization";

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return { ApiError: actual.ApiError, api: {
    mediaOrganizationCatalog: vi.fn(), createMediaCollection: vi.fn(), createMediaTag: vi.fn(),
  } };
});

function Management() {
  const organization = useMediaOrganization(vi.fn());
  return <><button onClick={() => organization.setManage(true)}>Manage choices</button>
    {organization.manage && <MediaOrganizationManager organization={organization} />}</>;
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.mediaOrganizationCatalog).mockResolvedValue({ items: [], next_cursor: null, revision: 1 });
});

it.each([
  ["albums", 422, "media-collection-invalid"],
  ["tags", 422, "media-tag-invalid"],
  ["tags", 409, "media-tag-conflict"],
] as const)("allows editing a refused %s name after %s %s", async (kind, status, code) => {
  const album = kind === "albums";
  const method = album ? vi.mocked(api.createMediaCollection) : vi.mocked(api.createMediaTag);
  method.mockRejectedValueOnce(new ApiError(status, null, "The request was refused.", code));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><Management /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Manage choices" }));
  const name = screen.getByRole("textbox", { name: album ? "New album name" : "New tag label" });
  fireEvent.change(name, { target: { value: "Studies" } });
  fireEvent.click(screen.getByRole("button", { name: album ? "Create album" : "Create tag" }));
  await screen.findByText("Creation was refused. Edit the name and try again.");
  const firstKey = method.mock.calls[0][2];
  expect(name).not.toHaveAttribute("readonly");
  fireEvent.click(screen.getByRole("button", { name: "Close albums and tags" }));
  fireEvent.click(screen.getByRole("button", { name: "Manage choices" }));
  const reopened = screen.getByRole("textbox", { name: album ? "New album name" : "New tag label" });
  expect(reopened).not.toHaveAttribute("readonly");
  fireEvent.change(reopened, { target: { value: "New studies" } });
  method.mockResolvedValueOnce(album
    ? { id: `collection_${"a".repeat(32)}`, kind: "manual", name: "New studies", description: "", version: 1 }
    : { id: `mediatag_${"b".repeat(32)}`, slug: "new-studies", label: "New studies", color: null, version: 1 });
  fireEvent.click(screen.getByRole("button", { name: album ? "Create album" : "Create tag" }));
  await waitFor(() => expect(reopened).toHaveValue(""));
  expect(method.mock.calls[1][2]).toMatch(/^[0-9a-f]{32}$/);
  expect(method.mock.calls[1][2]).not.toBe(firstKey);
});
afterEach(cleanup);

it.each(["albums", "tags"] as const)("keeps an uncertain %s creation across closing and reopening", async (kind) => {
  const album = kind === "albums";
  const method = album ? vi.mocked(api.createMediaCollection) : vi.mocked(api.createMediaTag);
  const result = album
    ? { id: `collection_${"a".repeat(32)}`, kind: "manual", name: "Studies", description: "", version: 1 }
    : { id: `mediatag_${"b".repeat(32)}`, slug: "studies", label: "Studies", color: null, version: 1 };
  method.mockRejectedValueOnce(new Error("connection-reset")).mockResolvedValueOnce(result);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><Management /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Manage choices" }));
  fireEvent.change(screen.getByRole("textbox", { name: album ? "New album name" : "New tag label" }), { target: { value: "Studies" } });
  fireEvent.click(screen.getByRole("button", { name: album ? "Create album" : "Create tag" }));
  await screen.findByText("Creation could not be confirmed. Try again with the same name; the original request is kept when you close this window.");
  const first = method.mock.calls[0];
  expect(first[2]).toMatch(/^[0-9a-f]{32}$/);
  fireEvent.click(screen.getByRole("button", { name: "Close albums and tags" }));
  expect(screen.queryByRole("dialog")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Manage choices" }));
  const name = screen.getByRole("textbox", { name: album ? "New album name" : "New tag label" });
  expect(name).toHaveValue("Studies");
  expect(name).toHaveAttribute("readonly");
  fireEvent.click(screen.getByRole("button", { name: album ? "Try creating this album again" : "Try creating this tag again" }));
  await waitFor(() => expect(name).toHaveValue(""));
  expect(method).toHaveBeenCalledTimes(2);
  expect(method.mock.calls[1]).toEqual(first);
  expect(name).not.toHaveAttribute("readonly");
});
