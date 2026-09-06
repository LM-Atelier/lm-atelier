import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import type { WorkerStatus } from "./types";
import { workerFailureSummary } from "./workerFailures";

function troubleshootingPage(): string {
  // Walk up rather than assume a working directory: vitest runs from the web
  // workspace, the repository root is two levels above it, and a wrong guess
  // would silently pass an empty document.
  for (const relative of ["../../docs", "../docs", "docs"]) {
    const candidate = resolve(process.cwd(), relative, "TROUBLESHOOTING.md");
    if (existsSync(candidate)) return readFileSync(candidate, "utf8");
  }
  throw new Error("TROUBLESHOOTING.md was not found from " + process.cwd());
}

const TROUBLESHOOTING = troubleshootingPage();

function worker(overrides: Partial<WorkerStatus>): WorkerStatus {
  return {
    name: "chat",
    state: "exited",
    managed: true,
    running: false,
    active_jobs: 0,
    queued_jobs: 0,
    ...overrides,
  } as WorkerStatus;
}

describe("workerFailureSummary", () => {
  it("leads with what happened rather than with an exit code", () => {
    const summary = workerFailureSummary(
      worker({
        failure_code: "oom_vram",
        failure_detail: "chat worker exited with code 1.",
      }),
    );
    expect(summary).toContain("graphics memory");
    expect(summary).not.toContain("exited with code");
  });

  it("names the worker the failure belongs to", () => {
    expect(workerFailureSummary(worker({ name: "media", failure_code: "port_in_use" })))
      .toContain("media");
  });

  it("falls back to the raw detail when the failure was not recognised", () => {
    const summary = workerFailureSummary(
      worker({ failure_code: null, failure_detail: "chat worker exited with code 9." }),
    );
    expect(summary).toBe("chat worker exited with code 9.");
  });

  it("never leaves a failed worker with an empty message", () => {
    expect(workerFailureSummary(worker({}))).toBe("chat worker stopped unexpectedly.");
  });

  // The same rule the setup-readiness docs test enforces: someone reads a
  // sentence in the app and searches for it. If the two drift, the page is not
  // stale, it is unfindable - worse than not documenting the failure at all.
  it.each([
    ["oom_vram", "needs more graphics memory than this computer has free"],
    ["oom_host", "needs more system memory than this computer has free"],
    ["port_in_use", "port is already in use"],
    ["model_incompatible", "could not read the selected model"],
    ["executable_missing", "engine program could not be started"],
    ["startup_timeout", "took too long to start"],
    ["crashed", "stopped unexpectedly"],
  ])("documents the sentence shown for %s", (code, phrase) => {
    const summary = workerFailureSummary(
      worker({ name: "chat", failure_code: code as WorkerStatus["failure_code"] }),
    );
    expect(summary).toContain(phrase);
    expect(TROUBLESHOOTING).toContain(phrase);
  });

  it("ignores a code it does not know rather than rendering undefined", () => {
    const summary = workerFailureSummary(
      worker({
        failure_code: "unknown",
        failure_detail: "chat worker exited with code 3.",
      }),
    );
    expect(summary).toBe("chat worker exited with code 3.");
  });
});
