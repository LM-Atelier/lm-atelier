import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { EditedBranchCards, type EditedBranchCardsProps } from "./EditedBranchCards";
import type { EditedBranch, Job, JobStatus, WorkStep } from "./types";

const stamp = "2026-09-07T12:00:00Z";
afterEach(cleanup);

function step(planId: string, ordinal: number, status: JobStatus = "queued"): WorkStep {
  return { id: `${planId}-step-${ordinal}`, plan_id: planId, run_id: `${planId}-run-${ordinal}`, ordinal,
    display_group: null, operation: "text_to_image", status, prompt: "A green landscape.", profile_id: null,
    workflow_revision_id: null, settings_json: {}, input_bindings_json: [], output_contract_json: [{ type: "image" }],
    queue_class: "media_compute", error: null, created_at: stamp, updated_at: stamp };
}

function job(workStep: WorkStep, position = 0): Job {
  return { id: `${workStep.id}-job`, kind: "image", status: workStep.status as JobStatus, run_id: workStep.run_id,
    work_plan_id: workStep.plan_id, work_step_id: workStep.id, progress: 0, phase: workStep.status,
    queue_resource: "media_compute", queue_ticket: "opaque-ticket", payload_json: {}, result_json: {},
    error: "Unrendered internal error", attempt: 0, cancellable: true, created_at: stamp, updated_at: stamp,
    started_at: null, completed_at: null,
    progress_json: { version: 2, stage: workStep.status, stage_progress: null, overall_progress: null,
      completed_units: null, total_units: null, unit: null, bytes_reused: 0, rate_bytes_per_second: null,
      eta_seconds: null, file_index: null, file_count: null, queue_resource: "media_compute", queue_position: position,
      queue_length: 5, blocked_by: [], indeterminate: true, updated_at: stamp } };
}

function branch(id = "edit-one", statuses: JobStatus[] = ["queued"]): EditedBranch {
  const steps = statuses.map((status, index) => step(id, index + 1, status));
  return { source_message_id: "original-user", source_run_id: "original-run", branch_head_message_id: `${id}-head`,
    source_available: true, can_continue: statuses.every((status) => status === "complete"),
    plan: { id, chat_id: "chat-one", idempotency_key: "request-one", source_action: "edit_and_branch",
      persistence_scope: "durable", status: statuses[0], context_head_message_id: null, transcript_sequence: 987,
      priority: 0, planner_version: "media-outputs-v1", failure_policy: "stop_dependents", summary_json: {}, steps,
      created_at: stamp, updated_at: stamp }, jobs: steps.map((item, index) => job(item, index * 3)) };
}

function props(branches: EditedBranch[]): EditedBranchCardsProps {
  return { branches, activeHeadId: "original-head", onView: vi.fn(), onContinue: vi.fn(), onCancelPlan: vi.fn(),
    onRetryPlan: vi.fn(), onCancelStep: vi.fn(), onRetryStep: vi.fn() };
}

it("shows resource queue progress using Next and ahead without exposing tickets, sequence or errors", () => {
  render(<EditedBranchCards {...props([branch("edit-one", ["queued", "queued"])])} />);
  expect(screen.getByText("media compute · queued · Next")).toBeVisible();
  expect(screen.getByText("media compute · queued · 3 ahead")).toBeVisible();
  expect(screen.getByText("Edited version queued")).toBeVisible();
  expect(screen.queryByText(/opaque-ticket|987|Unrendered internal error/)).not.toBeInTheDocument();
});

it("views a queued branch without activating it and disables unavailable continuation", () => {
  const edited = branch();
  const handlers = props([edited]);
  render(<EditedBranchCards {...handlers} />);
  fireEvent.click(screen.getByRole("button", { name: "View edited branch" }));
  expect(handlers.onView).toHaveBeenCalledWith(edited);
  expect(screen.getByRole("button", { name: "Continue from this version" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Continue from this version" }));
  expect(handlers.onContinue).not.toHaveBeenCalled();
});

it("continues only the explicitly selected available alternate and disables the active version", () => {
  const first = branch("first", ["complete"]);
  const second = branch("second", ["complete"]);
  const handlers = props([first, second]);
  render(<EditedBranchCards {...handlers} activeHeadId={first.branch_head_message_id} />);
  const cards = screen.getAllByRole("article");
  expect(within(cards[0]).getByRole("button", { name: "Continue from this version" })).toBeDisabled();
  fireEvent.click(within(cards[1]).getByRole("button", { name: "Continue from this version" }));
  expect(handlers.onContinue).toHaveBeenCalledTimes(1);
  expect(handlers.onContinue).toHaveBeenCalledWith(second);
  expect(handlers.onView).not.toHaveBeenCalled();
});

it("targets plan and step controls locally while unrelated versions remain running", () => {
  const edited = branch("edited", ["queued", "failed"]);
  const handlers = props([branch("unrelated", ["running"]), edited]);
  render(<EditedBranchCards {...handlers} />);
  const card = within(screen.getAllByRole("article")[1]);
  fireEvent.click(card.getByRole("button", { name: "Cancel edited version" }));
  fireEvent.click(card.getByRole("button", { name: "Retry unsuccessful outputs" }));
  fireEvent.click(card.getByRole("button", { name: "Cancel output 1" }));
  fireEvent.click(card.getByRole("button", { name: "Retry output 2" }));
  expect(handlers.onCancelPlan).toHaveBeenCalledTimes(1);
  expect(handlers.onCancelPlan).toHaveBeenCalledWith("edited");
  expect(handlers.onRetryPlan).toHaveBeenCalledTimes(1);
  expect(handlers.onRetryPlan).toHaveBeenCalledWith("edited");
  expect(handlers.onCancelStep).toHaveBeenCalledTimes(1);
  expect(handlers.onCancelStep).toHaveBeenCalledWith("edited-step-1");
  expect(handlers.onRetryStep).toHaveBeenCalledTimes(1);
  expect(handlers.onRetryStep).toHaveBeenCalledWith("edited-step-2");
});

it("withholds cancellation for uncancellable work, including aggregate mixed availability", () => {
  const edited = branch("mixed", ["running", "queued"]);
  edited.jobs[0].cancellable = false;
  render(<EditedBranchCards {...props([edited])} />);
  expect(screen.queryByRole("button", { name: "Cancel edited version" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Cancel output 1" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Cancel output 2" })).toBeEnabled();
});

it("ignores jobs belonging to another plan or another step when offering controls", () => {
  const edited = branch("edited", ["queued", "failed"]);
  edited.jobs[0].work_plan_id = "other-plan";
  edited.jobs[1].work_step_id = "other-step";

  render(<EditedBranchCards {...props([edited])} />);
  expect(screen.queryByRole("button", { name: /Cancel|Retry/ })).not.toBeInTheDocument();
});

it.each(["failed", "cancelled", "interrupted"] as const)("offers retry for %s work but keeps a missing original explicit", (status) => {
  const edited = branch("edited", [status]);
  edited.source_available = false;
  render(<EditedBranchCards {...props([edited])} />);
  expect(screen.getByText("Original message unavailable")).toBeVisible();
  expect(screen.getByRole("button", { name: "Retry unsuccessful outputs" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "View edited branch" })).toBeEnabled();
  expect(screen.queryByRole("button", { name: "Cancel edited version" })).not.toBeInTheDocument();
});

it("renders nothing when there are no alternate versions", () => {
  const { container } = render(<EditedBranchCards {...props([])} />);
  expect(container).toBeEmptyDOMElement();
});

it("serializes head selection while keeping preview available", () => {
  const first = branch("first", ["complete"]);
  const second = branch("second", ["complete"]);
  const handlers = props([first, second]);
  render(<EditedBranchCards {...handlers} activatingPlanId={first.plan.id} />);
  const cards = screen.getAllByRole("article");
  const pending = within(cards[0]).getByRole("button", { name: "Continue from this version" });
  expect(pending).toBeDisabled();
  fireEvent.click(pending);
  expect(handlers.onContinue).not.toHaveBeenCalled();
  expect(within(cards[1]).getByRole("button", { name: "Continue from this version" })).toBeDisabled();
  fireEvent.click(within(cards[1]).getByRole("button", { name: "View edited branch" }));
  expect(handlers.onView).toHaveBeenCalledWith(second);
  expect(handlers.onContinue).not.toHaveBeenCalled();
});