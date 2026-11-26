/** Unloading both workers' models from Settings, and saying why it was refused. */

import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import type { WorkerStatus } from "./types";
import { UnloadModelsButton } from "./UnloadModelsButton";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const UNLOAD = "/api/workers/unload";

function stopped(name: WorkerStatus["name"]): WorkerStatus {
  return {
    name,
    state: "stopped",
    managed: true,
    running: false,
    pid: null,
    profile_id: null,
    command: [],
    exit_code: null,
    estimated_memory_bytes: null,
    current_memory_bytes: null,
    peak_memory_bytes: null,
    active_jobs: 0,
    queued_jobs: 0,
  };
}

function answer(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

/** The local service: a session, and whatever the unload is answered with. */
function serve(unload: () => Promise<Response>) {
  const server = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>((path) =>
    path === "/api/session" ? Promise.resolve(answer({ csrf_token: "neutral-csrf" })) : unload(),
  );
  vi.stubGlobal("fetch", server);
  return server;
}

function unloadRequests(server: ReturnType<typeof serve>) {
  return server.mock.calls.filter(([path]) => path === UNLOAD);
}

function show() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <UnloadModelsButton />
    </QueryClientProvider>,
  );
  return client;
}

describe("Unload all models", () => {
  it("unloads both workers in one request and names what loads them again", async () => {
    const workers = [stopped("chat"), stopped("media")];
    const server = serve(async () => answer(workers));
    const client = show();

    fireEvent.click(screen.getByRole("button", { name: "Unload all models" }));

    // The controls that load a model once both are stopped: the Workers
    // section has no chat load then, so it is not named.
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Models unloaded. The next message or generation loads what it needs, and so do Load on a chat profile, Start ComfyUI and restarting LM Atelier.",
    );
    const requests = unloadRequests(server);
    expect(requests).toHaveLength(1);
    const [, init] = requests[0];
    expect(init?.method).toBe("POST");
    expect(new Headers(init?.headers).get("x-local-lm-csrf")).toBe("neutral-csrf");
    // The worker cards read the statuses it answered with.
    expect(client.getQueryData(["workers"])).toEqual(workers);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each([
    [
      "workers-busy",
      "1 job is running or ready to start, so the models stay loaded.",
      "Work is running or ready to start, so the models stay loaded. Let it finish, or pause its category under Accepted work in the sidebar, then try again.",
    ],
    [
      "media-worker-in-use",
      "A ComfyUI workflow editor is open, so ComfyUI was left running and no models were released.",
      "ComfyUI is in use from its own page or a workflow editor, so it was left running. Finish that work or close the editor, then try again.",
    ],
  ])("explains a %s refusal in plain words", async (code, detail, shown) => {
    serve(async () => answer({ code, detail }, 409));
    show();

    fireEvent.click(screen.getByRole("button", { name: "Unload all models" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(shown);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("shows any other refusal as the service words it", async () => {
    serve(async () =>
      answer({ code: "worker-stop-failed", detail: "Worker shutdown did not complete." }, 422),
    );
    show();

    fireEvent.click(screen.getByRole("button", { name: "Unload all models" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Worker shutdown did not complete.");
  });

  it("is marked unavailable and ignores presses while the request runs", async () => {
    let finish!: (response: Response) => void;
    const server = serve(() => new Promise<Response>((resolve) => { finish = resolve; }));
    show();
    const button = screen.getByRole("button", { name: "Unload all models" });

    // A second press in the same moment, before the pending state shows.
    fireEvent.click(button);
    fireEvent.click(button);
    const pending = await screen.findByRole("button", { name: "Unloading…" });
    expect(pending).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(pending);
    await vi.waitFor(() => expect(unloadRequests(server)).toHaveLength(1));
    expect(unloadRequests(server)).toHaveLength(1);

    finish(answer([stopped("chat"), stopped("media")]));

    expect(await screen.findByRole("button", { name: "Unload all models" })).toHaveAttribute(
      "aria-disabled",
      "false",
    );
    expect(unloadRequests(server)).toHaveLength(1);
  });
});
