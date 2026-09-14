import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { Workflow, WorkflowFamily, WorkflowInstallProgress } from "../apps/web/src/types";

const stamp = "2026-09-01T00:00:00Z";
const workflow: Workflow = {
  id: "installation-example", family_id: "example-collection", name: "Example image workflow",
  description: "A neutral workflow", operation: "text_to_image", current_revision_id: "example-r1",
  revisions: [{
    id: "example-r1", workflow_id: "installation-example", version: 1, engine: "mock",
    engine_version: null, trusted: true, created_at: stamp, dependency_contract_sha256: "b".repeat(64),
    api_graph_json: {}, ui_graph_json: {}, input_schema_json: {}, dependencies_json: {},
  }],
};

function progress(phase: WorkflowInstallProgress["phase"]): WorkflowInstallProgress {
  return {
    id: "example-offer", workflow_revision_id: "example-r1", phase,
    status: phase === "completed" ? "completed" : "queued", total_downloads: 2,
    completed_downloads: phase === "completed" ? 2 : 1,
    pending_downloads: phase === "downloading" ? 1 : 0,
    paused_downloads: 0, failed_downloads: 0, cancelled_downloads: 0, unavailable_downloads: 0,
    attention_code: phase === "needs_attention" ? "workflow-dependencies-need-selection" : null,
  };
}

function family(snapshot: WorkflowInstallProgress): WorkflowFamily {
  return {
    id: "example-collection", name: "Example collection", description: "", use_case: "", tags: [],
    enabled: true, archived: false, compatibility: false, created_at: stamp, updated_at: stamp, preferences: [],
    variants: [{
      id: workflow.id, variant_key: "image", name: workflow.name, operation: workflow.operation,
      current_revision_id: "example-r1", current_revision_version: 1, engine: "mock",
      capabilities: ["image"], trusted: true, readiness: "setup_required",
      readiness_reason: "activation_not_ready", setup_resolution: "attention_required",
      install_offer: null, install_progress: snapshot,
    }],
  };
}

async function openWorkflows(page: Page, width: number, firstVisit = true) {
  await page.goto("/");
  const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
  if (firstVisit) {
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
  }
  const navigation = page.getByRole("button", { name: "Workflows", exact: true });
  if (width < 700 && !await navigation.isVisible()) {
    await page.getByRole("button", { name: "Toggle navigation" }).click();
  }
  await navigation.click();
}

for (const width of [1280, 375]) {
  test("installation status survives reload and keeps refresh focus at " + width + "px", async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    let phase: WorkflowInstallProgress["phase"] = "downloading";
    let holdNextRead = false;
    let releaseRead: (() => void) | undefined;
    let installRequests = 0;
    await page.route("**/api/workflow-families?*", route => route.fulfill({ json: [family(progress(phase))] }));
    await page.route("**/api/workflow-summaries", route => route.fulfill({ json: [{
      id: workflow.id, family_id: workflow.family_id, name: workflow.name, description: workflow.description,
      operation: workflow.operation, current_revision_id: workflow.current_revision_id,
      revision_count: 1, created_at: stamp, updated_at: stamp,
    }] }));
    await page.route("**/api/workflows/installation-example", route => route.fulfill({ json: workflow }));
    await page.route("**/api/workflow-install-offers/example-offer/install", route => {
      installRequests += 1;
      return route.fulfill({ status: 500, json: { detail: "Unexpected installation request" } });
    });
    await page.route("**/api/workflow-install-offers/example-offer/progress", async route => {
      expect(route.request().method()).toBe("GET");
      if (holdNextRead) {
        holdNextRead = false;
        await new Promise<void>(resolve => { releaseRead = resolve; });
      }
      return route.fulfill({ json: progress(phase) });
    });
    await openWorkflows(page, width);
    const card = page.locator(".workflow-family-browser").getByRole("region", {
      name: "Installation for Example image workflow",
    });
    await expect(card.getByRole("status")).toHaveText("Downloading workflow files");
    const refresh = card.getByRole("button", { name: "Refresh installation status" });
    await expect(refresh).toBeEnabled();
    holdNextRead = true;
    await refresh.click();
    await expect(refresh).toHaveAttribute("aria-disabled", "true");
    await expect(refresh).toBeFocused();
    phase = "needs_attention";
    releaseRead?.();
    await expect(card.getByRole("status")).toHaveText("Installation needs attention");
    await expect(refresh).toBeFocused();
    await expect(card).toContainText("Choose the installed dependencies in workflow setup.");
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await testInfo.attach("workflow-installation-attention", {
      body: await page.screenshot({ fullPage: true }), contentType: "image/png",
    });
    await card.getByRole("button", { name: "Review workflow setup" }).click();
    await expect(page.getByRole("region", { name: "Workflow dependencies", exact: true })).toBeVisible();
    phase = "completed";
    await openWorkflows(page, width, false);
    await expect(card.getByRole("status")).toHaveText("Installation completed");
    await expect(card).toContainText("Review workflow setup to check whether this workflow can run.");
    await expect(card).not.toContainText("example-offer");
    await expect(card).not.toContainText("Ready to run");
    expect(installRequests).toBe(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  });
}
