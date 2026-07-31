import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api } from "./api";
import { useDraftClassification } from "./useDraftClassification";

function Probe({ text, hasPriorVisual }: { text: string; hasPriorVisual: boolean }) {
  const reuses = useDraftClassification("chat-1", text, "image", hasPriorVisual);
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
