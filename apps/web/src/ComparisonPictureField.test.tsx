import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ComparisonPictureField } from "./ComparisonPictureField";
import { api } from "./api";
import type { ComparisonSource } from "./generationComparison";
import type { ArtifactLibraryItem } from "./types";

vi.mock("./api", () => ({ api: { artifacts: vi.fn() } }));

const PICTURE: ArtifactLibraryItem = {
  id: "sha256:" + "d".repeat(64), sha256: "d".repeat(64), kind: "image", media_type: "image/png",
  size_bytes: 1, original_name: "Harbor", metadata_json: {},
  created_at: "2026-01-01T00:00:00Z", reference_count: 0, chat_ids: [], project_ids: [],
};

function show(value: ComparisonSource | null) {
  const change = vi.fn();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><ComparisonPictureField value={value} onChange={change} /></QueryClientProvider>);
  return change;
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  localStorage.clear();
});

it("starts from words, and changes one library picture once one is chosen", async () => {
  vi.mocked(api.artifacts).mockResolvedValue([PICTURE]);
  const change = show(null);
  expect(screen.getByRole("radio", { name: "Words only: make a new picture" })).toBeChecked();

  fireEvent.click(screen.getByRole("radio", { name: /A picture from the Media Library/ }));
  fireEvent.click(await screen.findByRole("button", { name: "Harbor" }));
  fireEvent.click(screen.getByRole("button", { name: "Change this picture" }));

  expect(change).toHaveBeenCalledWith({ id: PICTURE.id });
});

it("closing the picker without a picture leaves the comparison making pictures from words", async () => {
  vi.mocked(api.artifacts).mockResolvedValue([PICTURE]);
  const change = show(null);

  fireEvent.click(screen.getByRole("radio", { name: /A picture from the Media Library/ }));
  fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

  expect(change).not.toHaveBeenCalled();
  expect(screen.getByRole("radio", { name: "Words only: make a new picture" })).toBeChecked();
});

it("shows the chosen picture and goes back to words when asked", () => {
  const change = show({ id: PICTURE.id });
  expect(screen.getByText("Both choices change this picture and keep its size.")).toBeVisible();

  fireEvent.click(screen.getByRole("radio", { name: "Words only: make a new picture" }));

  expect(change).toHaveBeenCalledWith(null);
});
