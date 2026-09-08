import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api } from "./api";
import { useDraftClassification } from "./useDraftClassification";
import type { PriorTurnEditBinding } from "./types";

function Probe({ text, hasPriorVisual, editSource }: { text: string; hasPriorVisual: boolean; editSource?: PriorTurnEditBinding }) {
  const reuses = useDraftClassification("chat-1", text, "image", hasPriorVisual, editSource);
  return <div data-testid="answer">{String(reuses)}</div>;
}

function mount(text: string, hasPriorVisual: boolean) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Probe text={text} hasPriorVisual={hasPriorVisual} />
    </QueryClientProvider>,
  );
}

describe("useDraftClassification", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("reports what the server says rather than matching patterns locally", async () => {
    // "Make her top red" is one of the phrasings the browser's own copy of the
    // router's patterns used to miss.
    const classify = vi
      .spyOn(api, "classifyDraft")
      .mockResolvedValue({ references_prior_visual: true });

    mount("Make her top red", true);

    await waitFor(() => expect(screen.getByTestId("answer")).toHaveTextContent("true"));
    expect(classify).toHaveBeenCalledWith("chat-1", "Make her top red", "image");
  });

  it("asks nothing when there is no prior visual to reuse", async () => {
    const classify = vi
      .spyOn(api, "classifyDraft")
      .mockResolvedValue({ references_prior_visual: true });

    mount("Make her top red", false);

    await waitFor(() => expect(screen.getByTestId("answer")).toHaveTextContent("false"));
    expect(classify).not.toHaveBeenCalled();
  });

  it("asks nothing for an empty draft, which is the composer's resting state", async () => {
    const classify = vi
      .spyOn(api, "classifyDraft")
      .mockResolvedValue({ references_prior_visual: true });

    mount("   ", true);

    await waitFor(() => expect(screen.getByTestId("answer")).toHaveTextContent("false"));
    expect(classify).not.toHaveBeenCalled();
  });

  it("does not claim an edit while the first answer is still in flight", async () => {
    vi.spyOn(api, "classifyDraft").mockReturnValue(new Promise(() => undefined));

    mount("Make her top red", true);

    expect(screen.getByTestId("answer")).toHaveTextContent("false");
  });
});

it("binds source classification and never reuses an answer from a different source snapshot", async () => {
  const classify = vi.spyOn(api, "classifyDraft")
    .mockResolvedValueOnce({ references_prior_visual: true })
    .mockReturnValue(new Promise(() => undefined));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const binding = { source_message_id: "source-user", source_run_id: "source-run", source_snapshot_sha256: "a".repeat(64) };
  const view = (editSource: PriorTurnEditBinding) => <QueryClientProvider client={client}>
    <Probe text="Recolor the previous image" hasPriorVisual editSource={editSource} />
  </QueryClientProvider>;
  const rendered = render(view(binding));
  await waitFor(() => expect(screen.getByTestId("answer")).toHaveTextContent("true"));
  expect(classify).toHaveBeenCalledWith("chat-1", "Recolor the previous image", "image", binding);
  const changed = { ...binding, source_snapshot_sha256: "b".repeat(64) };
  rendered.rerender(view(changed));
  await waitFor(() => expect(classify).toHaveBeenCalledWith("chat-1", "Recolor the previous image", "image", changed));
  expect(screen.getByTestId("answer")).toHaveTextContent("false");
  rendered.unmount();
  client.clear();
  vi.restoreAllMocks();
});
