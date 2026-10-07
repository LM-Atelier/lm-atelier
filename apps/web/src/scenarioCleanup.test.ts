import { expect, test, vi } from "vitest";
import { withScenarioCleanup } from "../../../e2e/scenario-cleanup";

test("returns the scenario result after cleanup finishes exactly once", async () => {
  let finish!: () => void;
  const held = new Promise<void>((resolve) => { finish = resolve; });
  const cleanup = vi.fn(() => held);
  let returned = false;
  const value = { completed: true };
  const running = withScenarioCleanup(async () => value, cleanup).then((result) => {
    returned = true;
    return result;
  });
  await Promise.resolve();
  expect(cleanup).toHaveBeenCalledTimes(1);
  expect(returned).toBe(false);
  finish();
  expect(await running).toBe(value);
  expect(cleanup).toHaveBeenCalledTimes(1);
});

test("keeps the exact scenario error when cleanup succeeds", async () => {
  const primary = new Error("The constructed position assertion failed.");
  const cleanup = vi.fn(async () => {});
  await expect(withScenarioCleanup(async () => { throw primary; }, cleanup)).rejects.toBe(primary);
  expect(cleanup).toHaveBeenCalledTimes(1);
});

test("reports a cleanup failure after a successful scenario", async () => {
  const secondary = new Error("The constructed cleanup request failed.");
  const cleanup = vi.fn(async () => { throw secondary; });
  await expect(withScenarioCleanup(async () => "done", cleanup)).rejects.toBe(secondary);
  expect(cleanup).toHaveBeenCalledTimes(1);
});

test("reports the primary assertion first and attaches the cleanup error as its cause", async () => {
  const primary = new Error("The constructed position assertion failed.", { cause: new Error("Polling failed.") });
  primary.name = "AssertionError";
  const secondary = new Error("The constructed cleanup request failed.");
  const cleanup = vi.fn(async () => { throw secondary; });
  const combined: unknown = await withScenarioCleanup(async () => { throw primary; }, cleanup).catch((error: unknown) => error);
  expect(combined).toBeInstanceOf(AggregateError);
  if (!(combined instanceof AggregateError)) throw new Error("Expected both failures.");
  expect(combined.message).toBe(primary.message);
  expect(combined.name).toBe(primary.name);
  expect(combined.stack).toBe(primary.stack);
  expect(combined.cause).toBe(secondary);
  expect(combined.errors[0]).toBe(primary);
  expect(combined.errors[1]).toBe(secondary);
  expect(primary.cause).toBeInstanceOf(Error);
  expect(cleanup).toHaveBeenCalledTimes(1);
});

test("retains a thrown undefined value alongside a cleanup failure", async () => {
  const secondary = new Error("The constructed cleanup request failed.");
  const combined: unknown = await withScenarioCleanup(async () => { throw undefined; }, async () => { throw secondary; })
    .catch((error: unknown) => error);
  expect(combined).toBeInstanceOf(AggregateError);
  if (!(combined instanceof AggregateError)) throw new Error("Expected both failures.");
  expect(combined.errors).toEqual([undefined, secondary]);
  expect(combined.cause).toBe(secondary);
});

test("does not start cleanup while the scenario is still running", async () => {
  let finish!: () => void;
  const scenario = new Promise<void>((resolve) => { finish = resolve; });
  const cleanup = vi.fn(async () => {});
  const running = withScenarioCleanup(() => scenario, cleanup);
  await Promise.resolve();
  expect(cleanup).not.toHaveBeenCalled();
  finish();
  await running;
  expect(cleanup).toHaveBeenCalledTimes(1);
});
