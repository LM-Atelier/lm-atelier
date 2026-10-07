/** A model's tool waits for the Studio to know whether its workflow is here, and an exact edit never offers the model's Apply. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { useStudioImage } from "./useStudioImage";

vi.mock("./api", () => ({
  api: {
    openStudioSession: vi.fn(),
    studioSession: vi.fn(),
    favoriteArtifact: vi.fn(),
    artifact: vi.fn(),
    editTemplates: vi.fn(),
    studioCapabilities: vi.fn(),
  },
}));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

beforeEach(() => {
  const session = { id: "chat-studio", messages: [] } as never;
  vi.mocked(api.openStudioSession).mockResolvedValue(session);
  vi.mocked(api.studioSession).mockResolvedValue(session);
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  });
});

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

function studio(client = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
  return render(
    <QueryClientProvider client={client}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
}

it("holds a model's edit while the Studio checks whether its tool can run", async () => {
  vi.mocked(api.studioCapabilities).mockReturnValue(new Promise(() => undefined));
  studio();

  fireEvent.change(await screen.findByRole("textbox", { name: /Describe the edit/ }), {
    target: { value: "make the sky warmer" },
  });

  expect(screen.getByText("Checking whether this tool can run here…")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Apply edit" })).toHaveAttribute("aria-disabled", "true");

  // An edit made without a model has nothing to wait for.
  fireEvent.click(screen.getByRole("button", { name: /^Correct the perspective/ }));
  expect(screen.queryByText("Checking whether this tool can run here…")).toBeNull();
});

it("keeps holding when the check fails, and offers the edit once a second check answers", async () => {
  vi.mocked(api.studioCapabilities)
    .mockRejectedValueOnce(new Error("the server did not answer"))
    .mockResolvedValue({ tools: [] });
  studio();
  fireEvent.change(await screen.findByRole("textbox", { name: /Describe the edit/ }), {
    target: { value: "make the sky warmer" },
  });

  expect(await screen.findByText("Could not check whether this tool can run here.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Apply edit" })).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));

  await waitFor(() => expect(screen.getByRole("button", { name: "Apply edit" })).toHaveAttribute("aria-disabled", "false"));
  expect(screen.queryByText("Could not check whether this tool can run here.")).toBeNull();
});

it("keeps using the report it has when a later check fails", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
  const first = studio(client);
  await waitFor(() => expect(api.studioCapabilities).toHaveBeenCalledTimes(1));
  first.unmount();

  // Asked again on the next visit, and the server does not answer this time.
  vi.mocked(api.studioCapabilities).mockRejectedValueOnce(new Error("the server did not answer"));
  studio(client);
  fireEvent.change(await screen.findByRole("textbox", { name: /Describe the edit/ }), {
    target: { value: "make the sky warmer" },
  });

  await waitFor(() => expect(api.studioCapabilities).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(screen.getByRole("button", { name: "Apply edit" })).toHaveAttribute("aria-disabled", "false"));
  expect(screen.queryByText("Could not check whether this tool can run here.")).toBeNull();
});

it("offers no model's Apply beside a perspective correction, which applies from its own panel", async () => {
  vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
  studio();

  fireEvent.click(await screen.findByRole("button", { name: /^Correct the perspective/ }));

  expect(screen.getByRole("button", { name: "Apply the correction" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Apply edit" })).toBeNull();
});
