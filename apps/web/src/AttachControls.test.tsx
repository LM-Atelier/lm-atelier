import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AttachControls } from "./AttachControls";
import { api } from "./api";

vi.mock("./api", () => ({ api: { artifacts: vi.fn() } }));

function renderControls(onAttach = vi.fn(), onPickFile = vi.fn()) {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <AttachControls disabled={false} onPickFile={onPickFile} onAttach={onAttach} />
    </QueryClientProvider>,
  );
  return { onAttach, onPickFile };
}

describe("AttachControls", () => {
  afterEach(cleanup);

  it("offers both ways to attach, each one click", () => {
    // Folding these behind a chooser would add a step to the common case.
    vi.mocked(api.artifacts).mockResolvedValue([]);
    const { onPickFile } = renderControls();

    fireEvent.click(screen.getByRole("button", { name: "Attach file" }));

    expect(onPickFile).toHaveBeenCalledOnce();
    expect(screen.getByRole("button", { name: "Attach from the library" })).toBeInTheDocument();
  });

  it("attaches a library picture as what it already is, not as an upload", async () => {
    vi.mocked(api.artifacts).mockResolvedValue([
      {
        id: "artifact-1",
        sha256: "abc",
        kind: "image",
        media_type: "image/png",
        size_bytes: 1,
        original_name: "A generated picture",
        metadata_json: {},
        created_at: "2026-08-05T00:00:00Z",
        favorite: false,
        reference_count: 0,
        chat_ids: [],
        project_ids: [],
      },
    ] as never);
    const { onAttach } = renderControls();

    fireEvent.click(screen.getByRole("button", { name: "Attach from the library" }));
    fireEvent.click(await screen.findByRole("button", { name: "A generated picture" }));
    // The dialog's confirm button, which names how many are chosen. "Attach
    // file" is still on screen behind it, so this is matched exactly.
    fireEvent.click(screen.getByRole("button", { name: "Attach 1" }));

    // The transcript must not claim a person supplied what the app produced.
    expect(onAttach).toHaveBeenCalledWith(
      expect.objectContaining({ id: "artifact-1", origin: "generated" }),
    );
  });
});
