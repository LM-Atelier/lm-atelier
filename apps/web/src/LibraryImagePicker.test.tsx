import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { LibraryImagePicker } from "./LibraryImagePicker";
import { api } from "./api";
import type { ArtifactLibraryItem } from "./types";

vi.mock("./api", () => ({ api: { artifacts: vi.fn() } }));

function item(index: number): ArtifactLibraryItem {
  return {
    id: `image-${index}`, sha256: "a".repeat(64), kind: "image", media_type: "image/png",
    size_bytes: 1, original_name: `Image ${index}`, metadata_json: {},
    created_at: "2026-01-01T00:00:00Z", reference_count: 0, chat_ids: [], project_ids: [],
  };
}

const firstPage = Array.from({ length: 50 }, (_, index) => item(index));

function show() {
  const confirm = vi.fn();
  const close = vi.fn();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = render(
    <QueryClientProvider client={client}>
      <LibraryImagePicker title="Choose images" confirmLabel="Attach" onConfirm={confirm} onClose={close} />
    </QueryClientProvider>,
  );
  return { ...view, confirm, close };
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("loads bounded pages and preserves selection order across them", async () => {
  vi.mocked(api.artifacts).mockResolvedValueOnce(firstPage).mockResolvedValueOnce([item(49), item(50)]);
  const view = show();
  fireEvent.click(await screen.findByRole("button", { name: "Image 4" }));
  expect(api.artifacts).toHaveBeenNthCalledWith(
    1, "image", "", false, { limit: 50, offset: 0 }, expect.any(AbortSignal),
  );
  fireEvent.click(screen.getByRole("button", { name: "Load more images" }));
  fireEvent.click(await screen.findByRole("button", { name: "Image 50" }));
  expect(screen.getAllByRole("button", { name: "Image 49" })).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Image 1" }));
  expect(api.artifacts).toHaveBeenNthCalledWith(
    2, "image", "", false, { limit: 50, offset: 50 }, expect.any(AbortSignal),
  );
  expect(screen.queryByRole("button", { name: "Load more images" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Attach 3" }));
  expect(view.confirm).toHaveBeenCalledWith([item(4), item(50), item(1)]);
  expect(view.close).toHaveBeenCalledOnce();
});

it("retains selected images when another page fails and retries that page", async () => {
  vi.mocked(api.artifacts).mockResolvedValueOnce(firstPage)
    .mockRejectedValueOnce(new Error("Cannot read more images."))
    .mockResolvedValueOnce([item(50)]);
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Image 4" }));
  fireEvent.click(screen.getByRole("button", { name: "Load more images" }));
  expect(await screen.findByText("Cannot read more images.")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Image 4" }).getAttribute("aria-pressed")).toBe("true");
  fireEvent.click(screen.getByRole("button", { name: "Retry loading images" }));
  expect(await screen.findByRole("button", { name: "Image 50" })).toBeTruthy();
  expect(api.artifacts).toHaveBeenNthCalledWith(
    3, "image", "", false, { limit: 50, offset: 50 }, expect.any(AbortSignal),
  );
});

it("cancels the in-flight page when the picker closes", async () => {
  vi.mocked(api.artifacts).mockImplementation(() => new Promise(() => {}));
  const view = show();
  await waitFor(() => expect(api.artifacts).toHaveBeenCalledWith(
    "image", "", false, { limit: 50, offset: 0 }, expect.any(AbortSignal),
  ));
  const signal = vi.mocked(api.artifacts).mock.calls[0][4];
  view.unmount();
  expect(signal?.aborted).toBe(true);
});
