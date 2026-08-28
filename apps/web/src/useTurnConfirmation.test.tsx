import { useEffect } from "react";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import type { TurnConfirmationHandler } from "./api";
import { useTurnConfirmation } from "./useTurnConfirmation";

function Harness({ onReady }: { onReady: (confirm: TurnConfirmationHandler) => void }) {
  const [dialog, confirm] = useTurnConfirmation();
  useEffect(() => onReady(confirm), [confirm, onReady]);
  return dialog;
}

afterEach(cleanup);

it("renders ordered-plan facts and resolves cancel without acting", async () => {
  let requestConfirmation!: TurnConfirmationHandler;
  const capture = (confirm: TurnConfirmationHandler) => { requestConfirmation = confirm; };
  render(<Harness onReady={capture} />);
  let answer!: Promise<boolean>;
  act(() => {
    answer = requestConfirmation({
      kind: "ordered_plan",
      title: "Start ordered plan?",
      question: "This request will run 3 steps in sequence.",
      confirmLabel: "Start plan",
      details: {
        sequence: ["text", "image", "video"],
        videoDurationSeconds: 4,
        estimatedWorkingBytes: 2 * 1024 ** 3,
      },
    });
  });

  expect(screen.getByText("Confirm action")).toBeInTheDocument();
  expect(screen.getByText("Sequence: text \u2192 image \u2192 video")).toBeVisible();
  expect(screen.getByText("Video duration: about 4 seconds")).toBeVisible();
  expect(screen.getByText("Working space: up to 2.0 GB")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await expect(answer).resolves.toBe(false);
});

it("renders media-route facts and resolves only the explicit action", async () => {
  let requestConfirmation!: TurnConfirmationHandler;
  const capture = (confirm: TurnConfirmationHandler) => { requestConfirmation = confirm; };
  render(<Harness onReady={capture} />);
  let answer!: Promise<boolean>;
  act(() => {
    answer = requestConfirmation({
      kind: "media_route",
      title: "Start video generation?",
      question: "Auto mode suggests a video generation.",
      confirmLabel: "Start video",
      details: {
        operation: "video",
        durationSeconds: 6,
        estimatedIntermediateBytes: 3 * 1024 ** 3,
      },
    });
  });

  expect(screen.getByText("Suggested operation: video")).toBeVisible();
  expect(screen.getByText("Output duration: about 6 seconds")).toBeVisible();
  expect(screen.getByText("Intermediate space: up to 3.0 GB")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Start video" }));
  await expect(answer).resolves.toBe(true);
});
