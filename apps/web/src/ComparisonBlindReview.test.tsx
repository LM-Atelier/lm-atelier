import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { ComparisonBlindReview } from "./ComparisonBlindReview";
import type { BlindPosition, GenerationExperiment, GenerationExperimentBlindView } from "./generationExperimentTypes";
import { SENSITIVE_MEDIA_KEY } from "./sensitiveMedia";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, openBlindView: vi.fn(), blindView: vi.fn(), sayBlindPreference: vi.fn() } };
});

const KEY = "lm-atelier.generation-comparison.blind-view";
const EXPERIMENT = {
  id: "gexp-1", state: "started", evaluation_mode: "blind", blind_pending: true, evaluation: null,
  arms: [{ id: "arm-1", ordinal: 1, label: "Fewer steps", trials: [] }, { id: "arm-2", ordinal: 2, label: "More steps", trials: [] }],
} as unknown as GenerationExperiment;

function view(id: string, positions: Partial<BlindPosition>[] = [{}, {}]): GenerationExperimentBlindView {
  return {
    id, experiment_id: "gexp-1", evaluation: null, reveal: null,
    positions: positions.map((position, index) => ({ position: index + 1, status: "complete", ready: true, ...position })),
  };
}

/** The review as the results page shows it, beside the page's own read of the comparison. */
function Page({ reread }: { reread: () => Promise<GenerationExperiment> }) {
  useQuery({ queryKey: ["generation-experiments", "gexp-1"], queryFn: reread });
  return <ComparisonBlindReview experiment={EXPERIMENT} />;
}

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const reread = vi.fn(async () => EXPERIMENT);
  render(<QueryClientProvider client={client}><Page reread={reread} /></QueryClientProvider>);
  return reread;
}

beforeEach(() => {
  window.sessionStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  window.sessionStorage.clear();
  localStorage.clear();
});

it("opens a viewing and shows the pictures only by where they are shown", async () => {
  vi.mocked(api.openBlindView).mockResolvedValue(view("gview-1"));
  show();

  const section = await screen.findByRole("region", { name: "Compared blind" });

  const pictures = within(section).getAllByRole("img");
  expect(pictures.map((picture) => picture.getAttribute("alt"))).toEqual(["Shown at position 1", "Shown at position 2"]);
  expect(pictures.map((picture) => picture.getAttribute("src"))).toEqual([
    "/api/generation-experiments/gexp-1/blind-views/gview-1/pictures/1",
    "/api/generation-experiments/gexp-1/blind-views/gview-1/pictures/2",
  ]);
  // No choice is named anywhere in it, in its text or its markup.
  for (const label of ["Fewer steps", "More steps", "arm-1", "arm-2"]) expect(section.outerHTML).not.toContain(label);
  expect(within(section).getByRole("status")).toHaveTextContent("Both pictures are ready");
  expect(JSON.parse(window.sessionStorage.getItem(KEY) ?? "null")).toEqual({ experimentId: "gexp-1", viewId: "gview-1" });
});

it("keeps showing the same viewing after a reload", async () => {
  window.sessionStorage.setItem(KEY, JSON.stringify({ experimentId: "gexp-1", viewId: "gview-kept" }));
  vi.mocked(api.blindView).mockResolvedValue(view("gview-kept"));
  show();

  expect(await screen.findByRole("img", { name: "Shown at position 2" })).toHaveAttribute("src", expect.stringContaining("gview-kept"));
  expect(api.blindView).toHaveBeenCalledWith("gexp-1", "gview-kept", expect.anything());
  expect(api.openBlindView).not.toHaveBeenCalled();
});

it("opens a new viewing when the kept one is gone or belongs to another comparison", async () => {
  window.sessionStorage.setItem(KEY, JSON.stringify({ experimentId: "gexp-1", viewId: "gview-gone" }));
  vi.mocked(api.blindView).mockRejectedValue(new ApiError(404, "This blind viewing no longer exists.", "This blind viewing no longer exists.",
    "generation-experiment-blind-view-not-found"));
  vi.mocked(api.openBlindView).mockResolvedValue(view("gview-new"));
  show();

  expect(await screen.findByRole("img", { name: "Shown at position 1" })).toHaveAttribute("src", expect.stringContaining("gview-new"));
  expect(api.openBlindView).toHaveBeenCalledExactlyOnceWith("gexp-1");
});

it("waits for a picture before it can be preferred", async () => {
  vi.mocked(api.openBlindView).mockResolvedValue(view("gview-1", [{ status: "running", ready: false }, { status: "queued", ready: false }]));
  show();

  expect(await screen.findByText("0 of 2 pictures ready")).toBeInTheDocument();
  expect(screen.queryByRole("img")).toBeNull();
  expect(screen.getByText("Being made")).toBeInTheDocument();
  for (const name of ["Prefer picture 1", "Prefer picture 2", "Tie", "Neither suits"]) {
    const button = screen.getByRole("button", { name });
    expect(button).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(button);
  }
  expect(api.sayBlindPreference).not.toHaveBeenCalled();
});

it("says the preference by position, then shows the comparison with its choices named", async () => {
  vi.mocked(api.openBlindView).mockResolvedValue(view("gview-1"));
  vi.mocked(api.sayBlindPreference).mockResolvedValue({
    ...view("gview-1"),
    reveal: [{ position: 1, arm_ordinal: 2, label: "More steps" }, { position: 2, arm_ordinal: 1, label: "Fewer steps" }],
  });
  const reread = show();
  await waitFor(() => expect(reread).toHaveBeenCalledTimes(1));

  fireEvent.click(await screen.findByRole("button", { name: "Prefer picture 2" }));

  await waitFor(() => expect(reread).toHaveBeenCalledTimes(2));
  expect(api.sayBlindPreference).toHaveBeenCalledExactlyOnceWith("gexp-1", "gview-1", { preference: "preferred", position: 2 });
  expect(window.sessionStorage.getItem(KEY)).toBeNull();
});

it("keeps blind pictures blurred until each is shown", async () => {
  localStorage.setItem(SENSITIVE_MEDIA_KEY, "blur");
  vi.mocked(api.openBlindView).mockResolvedValue(view("gview-1"));
  show();
  const section = await screen.findByRole("region", { name: "Compared blind" });

  expect(within(section).queryAllByRole("img")).toHaveLength(0);
  fireEvent.click(within(section).getAllByRole("button", { name: "Show picture" })[1]);

  expect(within(section).getAllByRole("img").map((picture) => picture.getAttribute("alt"))).toEqual(["Shown at position 2"]);
});
