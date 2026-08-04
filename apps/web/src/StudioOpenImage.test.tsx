/** The studio needs a door of its own. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { StudioOpenImage } from "./StudioOpenImage";
import { firstImage } from "./studioFiles";

vi.mock("./api", () => ({ api: { upload: vi.fn() } }));

function renderOpener(onOpened = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <StudioOpenImage onOpened={onOpened} />
    </QueryClientProvider>,
  );
  return onOpened;
}

const png = () => new File(["x"], "portrait.png", { type: "image/png" });

describe("opening an image in the studio", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("takes a dropped image and opens what it uploaded", async () => {
    vi.mocked(api.upload).mockResolvedValue({ id: "artifact-1" } as never);
    const onOpened = renderOpener();

    const zone = document.querySelector(".studio-open-image")!;
    fireEvent.drop(zone, { dataTransfer: { files: [png()] } });

    // A picture on disk is the obvious thing to want to edit, and until now
    // it had no path into the studio at all.
    await waitFor(() => expect(onOpened).toHaveBeenCalledWith("artifact-1"));
  });

  it("takes a chosen file too, for anyone not dragging", async () => {
    vi.mocked(api.upload).mockResolvedValue({ id: "artifact-2" } as never);
    const onOpened = renderOpener();

    fireEvent.change(screen.getByLabelText("Choose an image to edit"), {
      target: { files: [png()] },
    });

    await waitFor(() => expect(onOpened).toHaveBeenCalledWith("artifact-2"));
  });

  it("ignores anything that is not an image", () => {
    // Dropping a PDF on a picture editor is a mistake, not a request.
    expect(firstImage([new File(["x"], "notes.pdf", { type: "application/pdf" })])).toBeNull();
    expect(firstImage([])).toBeNull();
    expect(firstImage([new File(["x"], "a.txt", { type: "text/plain" }), png()])?.name).toBe(
      "portrait.png",
    );
  });
});
