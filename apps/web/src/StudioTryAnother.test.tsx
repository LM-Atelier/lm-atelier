/** Try another: offered for a model's result, it sends that result's edit again on the pictures it was given. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioTryAnother } from "./StudioTryAnother";
import { api } from "./api";
import type { ChatDetail, Run } from "./types";
import { studioSteps, useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: { run: vi.fn(), openStudioSession: vi.fn(), studioSession: vi.fn(), sendTurn: vi.fn(), upload: vi.fn() },
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.clearAllMocks();
});

function part(type: string, values: Record<string, unknown>) {
  return { id: `${type}-${JSON.stringify(values)}`, position: 0, type, text: null, artifact_id: null, metadata_json: {}, ...values };
}

/** A relight: the turn gave the picture and a light map, and a model made art-2 from them. */
const SESSION = {
  id: "chat-studio",
  messages: [
    { id: "turn-1", role: "user", parts: [part("text", { text: "relight it" }), part("image", { artifact_id: "art-1" }), part("image", { artifact_id: "light-map" })] },
    { id: "result-1", role: "assistant", parts: [part("image", { artifact_id: "art-2" }), part("generation_metadata", { metadata_json: { run_id: "run-7" } })] },
    { id: "turn-2", role: "user", parts: [part("text", { text: "Flip horizontally" }), part("image", { artifact_id: "art-2" })] },
    { id: "result-2", role: "assistant", parts: [part("image", { artifact_id: "art-3" })] },
  ],
} as unknown as ChatDetail;
const [ORIGINAL, RELIT, FLIPPED] = studioSteps(SESSION, "art-1");
const RUN = {
  id: "run-7",
  workflow_revision_id: "rev-relight",
  provenance_json: { resolved_settings: { relight: { direction: "left" }, seed: 42 } },
} as unknown as Run;

function tryAnother(step: typeof RELIT, busy = false) {
  const onTry = vi.fn();
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })}>
      <StudioTryAnother session={SESSION} current={step} busy={busy} onTry={onTry} />
    </QueryClientProvider>,
  );
  return onTry;
}

describe("Try another", () => {
  it("is offered for a model's result, and not for the original or an exact edit", () => {
    tryAnother(ORIGINAL);
    expect(screen.queryByRole("button", { name: /Try another/ })).toBeNull();
    cleanup();
    tryAnother(FLIPPED);
    expect(screen.queryByRole("button", { name: /Try another/ })).toBeNull();
    cleanup();
    tryAnother(RELIT);
    expect(screen.getByRole("button", { name: /Try another/ })).toBeInTheDocument();
  });

  it("reads the result's run and sends its edit again on every picture it was given, with a new seed", async () => {
    vi.mocked(api.run).mockResolvedValue(RUN);
    const onTry = tryAnother(RELIT);

    fireEvent.click(screen.getByRole("button", { name: /Try another/ }));

    await waitFor(() => expect(onTry).toHaveBeenCalledTimes(1));
    expect(api.run).toHaveBeenCalledWith("run-7");
    expect(onTry).toHaveBeenCalledWith({
      words: "relight it",
      inputs: ["art-1", "light-map"],
      settings: { relight: { direction: "left" }, seed: -1 },
      workflowRevisionId: "rev-relight",
    });
  });

  it("says so and sends nothing when the run cannot be read", async () => {
    vi.mocked(api.run).mockRejectedValue(new Error("gone"));
    const onTry = tryAnother(RELIT);

    fireEvent.click(screen.getByRole("button", { name: /Try another/ }));

    expect(await screen.findByText("This result's edit could not be read, so nothing was sent.")).toBeInTheDocument();
    expect(onTry).not.toHaveBeenCalled();
  });

  it("waits while another edit is arriving", () => {
    tryAnother(RELIT, true);

    fireEvent.click(screen.getByRole("button", { name: /Try another/ }));

    expect(screen.getByRole("button", { name: /Try another/ })).toHaveAttribute("aria-disabled", "true");
    expect(api.run).not.toHaveBeenCalled();
  });
});

describe("sending an edit again", () => {
  it("sends the words, every picture, the settings and the workflow as one turn", async () => {
    vi.mocked(api.openStudioSession).mockResolvedValue(SESSION as never);
    vi.mocked(api.studioSession).mockResolvedValue(SESSION as never);
    vi.mocked(api.sendTurn).mockResolvedValue({} as never);
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{children}</QueryClientProvider>
    );
    const { result } = renderHook(() => useStudioSession("art-1", null), { wrapper });
    await waitFor(() => expect(result.current.sessionId).toBe("chat-studio"));

    result.current.again({
      words: "relight it",
      inputs: ["art-1", "light-map"],
      settings: { relight: { direction: "left" }, seed: -1 },
      workflowRevisionId: "rev-relight",
    });

    await waitFor(() => expect(api.sendTurn).toHaveBeenCalledTimes(1));
    expect(api.sendTurn).toHaveBeenCalledWith(
      "chat-studio",
      "relight it",
      "image",
      ["art-1", "light-map"],
      { relight: { direction: "left" }, seed: -1 },
      undefined,
      undefined,
      "rev-relight",
    );
    expect(api.upload).not.toHaveBeenCalled();
  });
});
