import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { ComparisonPreference } from "./ComparisonPreference";
import type { ExperimentEvaluation, GenerationExperiment } from "./generationExperimentTypes";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, evaluateGenerationExperiment: vi.fn() } };
});

function experiment(evaluation: ExperimentEvaluation | null = null, secondStatus = "complete"): GenerationExperiment {
  return {
    id: "gexp-1", state: "started", evaluation,
    arms: [
      { id: "arm-1", ordinal: 1, label: "Fewer steps", trials: [{ id: "trial-1", status: "complete" }] },
      { id: "arm-2", ordinal: 2, label: "More steps", trials: [{ id: "trial-2", status: secondStatus }] },
    ],
  } as unknown as GenerationExperiment;
}

function said(preference: ExperimentEvaluation["preference"], arm_ordinal: number | null = null): ExperimentEvaluation {
  return { preference, arm_ordinal, mode: "unblinded", note: null, created_at: "2026-10-01T00:00:00Z" };
}

/** The control as the results page shows it: reading the comparison from the cache the answer updates. */
function Shown() {
  const read = useQuery({ queryKey: ["generation-experiments", "gexp-1"], queryFn: () => experiment(), staleTime: Infinity });
  return read.data ? <ComparisonPreference experiment={read.data} /> : null;
}

function show(initial = experiment()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  client.setQueryData(["generation-experiments", "gexp-1"], initial);
  render(<QueryClientProvider client={client}><Shown /></QueryClientProvider>);
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("keeps the choice preferred and shows it as the answer", async () => {
  vi.mocked(api.evaluateGenerationExperiment).mockResolvedValue(experiment(said("preferred", 2)));
  show();
  expect(screen.getByRole("group", { name: "Which do you prefer?" })).toBeInTheDocument();
  expect(screen.queryByRole("status")).toBeNull();

  fireEvent.click(screen.getByRole("button", { name: "Prefer More steps" }));

  expect(await screen.findByRole("status")).toHaveTextContent("You preferred More steps.");
  expect(api.evaluateGenerationExperiment).toHaveBeenCalledExactlyOnceWith("gexp-1", { preference: "preferred", arm_ordinal: 2 });
  expect(screen.getByRole("button", { name: "Prefer More steps" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("button", { name: "Prefer Fewer steps" })).toHaveAttribute("aria-pressed", "false");
});

it("does not offer to prefer a choice whose picture is not made yet", async () => {
  vi.mocked(api.evaluateGenerationExperiment).mockResolvedValue(experiment(said("tied"), "running"));
  show(experiment(null, "running"));
  const unmade = screen.getByRole("button", { name: "Prefer More steps" });
  expect(unmade).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByRole("button", { name: "Prefer Fewer steps" })).toHaveAttribute("aria-disabled", "false");

  fireEvent.click(unmade);
  expect(api.evaluateGenerationExperiment).not.toHaveBeenCalled();
  // A tie or neither suiting can still be said of the picture that is made.
  fireEvent.click(screen.getByRole("button", { name: "Tie" }));

  expect(await screen.findByText("You called it a tie.")).toBeInTheDocument();
  expect(api.evaluateGenerationExperiment).toHaveBeenCalledExactlyOnceWith("gexp-1", { preference: "tied" });
});

it("says a tie or neither suiting without naming a choice", async () => {
  vi.mocked(api.evaluateGenerationExperiment)
    .mockResolvedValueOnce(experiment(said("tied")))
    .mockResolvedValueOnce(experiment(said("unsuitable")));
  show();

  fireEvent.click(screen.getByRole("button", { name: "Tie" }));
  expect(await screen.findByText("You called it a tie.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Neither suits" }));
  expect(await screen.findByText("You said neither suits.")).toBeInTheDocument();

  expect(vi.mocked(api.evaluateGenerationExperiment).mock.calls.map((call) => call[1])).toEqual([
    { preference: "tied" }, { preference: "unsuitable" },
  ]);
  expect(screen.getByRole("button", { name: "Neither suits" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("button", { name: "Tie" })).toHaveAttribute("aria-pressed", "false");
});

it("shows what was said before, and why a new saying was refused", async () => {
  vi.mocked(api.evaluateGenerationExperiment).mockRejectedValue(
    new ApiError(409, "Make the pictures before saying which you prefer.", "Make the pictures before saying which you prefer.",
      "generation-experiment-not-started"),
  );
  show(experiment(said("preferred", 1)));
  expect(screen.getByRole("status")).toHaveTextContent("You preferred Fewer steps.");

  fireEvent.click(screen.getByRole("button", { name: "Tie" }));

  expect(await screen.findByText("Make the pictures before saying which you prefer.")).toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole("button", { name: "Tie" })).toHaveAttribute("aria-disabled", "false"));
  // The earlier answer stands.
  expect(screen.getByRole("status")).toHaveTextContent("You preferred Fewer steps.");
});
