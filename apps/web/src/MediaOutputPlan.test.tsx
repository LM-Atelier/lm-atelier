import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MediaOutputPlan } from "./MediaOutputPlan";
import type { WorkPlan, WorkStep } from "./types";

const stamp = "2026-08-21T12:00:00Z";

function step(ordinal: number, status: string, prompt: string): WorkStep {
  return {
    id: `step-${ordinal}`,
    plan_id: "plan-prompt-library",
    run_id: `run-${ordinal}`,
    ordinal,
    display_group: "media_outputs",
    operation: "text_to_image",
    status,
    prompt,
    profile_id: null,
    workflow_revision_id: "workflow-revision",
    settings_json: { seed: 100 + ordinal, batch_size: 1 },
    input_bindings_json: [],
    output_contract_json: [{
      slot: `output-${ordinal}`,
      type: "image",
      index: ordinal,
      count: 2,
    }],
    queue_class: "media_compute",
    error: status === "failed" ? "Generation failed." : null,
    created_at: stamp,
    updated_at: stamp,
  };
}

describe("Prompt Library work-plan card", () => {
  it("shows exact item prompts and aggregate plus per-item controls", () => {
    const onCancelPlan = vi.fn();
    const onRetryPlan = vi.fn();
    const onCancelStep = vi.fn();
    const onRetryStep = vi.fn();
    const plan: WorkPlan = {
      id: "plan-prompt-library",
      chat_id: "chat-one",
      idempotency_key: "queue-one",
      source_action: "prompt_library",
      persistence_scope: "durable",
      status: "queued",
      context_head_message_id: null,
      transcript_sequence: 1,
      priority: 0,
      planner_version: "prompt-template-v1",
      failure_policy: "stop_dependents",
      summary_json: {},
      steps: [
        step(1, "queued", "A private first reviewed prompt."),
        step(2, "failed", "A private second reviewed prompt."),
      ],
      created_at: stamp,
      updated_at: stamp,
    };

    render(
      <MediaOutputPlan
        plan={plan}
        onCancelPlan={onCancelPlan}
        onRetryPlan={onRetryPlan}
        onCancelStep={onCancelStep}
        onRetryStep={onRetryStep}
      />,
    );
    fireEvent.click(screen.getByText("2 Prompt Library outputs"));
    expect(screen.getByText("A private first reviewed prompt.")).toBeVisible();
    expect(screen.getByText("A private second reviewed prompt.")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Cancel remaining outputs" }));
    fireEvent.click(screen.getByRole("button", { name: "Retry unsuccessful outputs" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel output 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Retry output 2" }));
    expect(onCancelPlan).toHaveBeenCalledWith(plan.id);
    expect(onRetryPlan).toHaveBeenCalledWith(plan.id);
    expect(onCancelStep).toHaveBeenCalledWith("step-1");
    expect(onRetryStep).toHaveBeenCalledWith("step-2");
  });
});
