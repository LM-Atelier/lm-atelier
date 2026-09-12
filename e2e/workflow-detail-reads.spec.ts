import { expect, test } from "@playwright/test";
import type { Workflow, WorkflowSummary } from "../apps/web/src/types";

const stamp = "2026-09-01T00:00:00Z";
function detail(id: string): Workflow {
  return {
    id, family_id: null, name: "Example " + id, description: "Neutral workflow",
    operation: "text_to_image", current_revision_id: id + "-r2",
    revisions: [1, 2].map((version) => ({
      id: id + "-r" + version, workflow_id: id, version, engine: "mock",
      engine_version: null, trusted: false, created_at: stamp,
      api_graph_json: { selected_workflow: id, revision: version }, ui_graph_json: {},
      input_schema_json: {}, dependencies_json: {},
    })),
  };
}
function summary(id: string): WorkflowSummary {
  const value = detail(id);
  return {
    id, family_id: null, name: value.name, description: value.description,
    operation: value.operation, current_revision_id: value.current_revision_id,
    revision_count: 2, created_at: stamp, updated_at: stamp,
  };
}

for (const width of [1280, 375]) {
  test("workflow Library loads graphs only after selection at " + width + "px", async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    let bulkReads = 0;
    const detailReads: string[] = [];
    await page.route("**/api/workflows", async (route) => {
      bulkReads += 1;
      await route.fulfill({ status: 500, json: { detail: "Bulk graphs are forbidden in this test" } });
    });
    await page.route("**/api/workflow-families?*", (route) => route.fulfill({ json: [] }));
    await page.route("**/api/workflow-summaries", (route) => route.fulfill({
      json: [summary("a"), summary("b")],
    }));
    await page.route(/\/api\/workflows\/[ab]$/, async (route) => {
      const id = new URL(route.request().url()).pathname.split("/").at(-1)!;
      detailReads.push(id);
      await route.fulfill({ json: detail(id) });
    });
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
    await page.getByRole("button", { name: "Workflows", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Select a workflow" })).toBeVisible();
    expect(bulkReads).toBe(0);
    expect(detailReads).toEqual([]);
    await page.getByRole("button", { name: /Example a/ }).click();
    await expect(page.getByRole("heading", { name: "Example a", exact: true })).toBeVisible();
    await expect(page.getByRole("combobox", { name: "Revision", exact: true })).toHaveValue("a-r2");
    await page.getByRole("combobox", { name: "Revision", exact: true }).selectOption("a-r1");
    await expect(page.getByRole("button", { name: "Restore as new revision" })).toBeVisible();
    expect(detailReads).toEqual(["a"]);
    await page.getByRole("button", { name: /Example b/ }).click();
    await expect(page.getByRole("heading", { name: "Example b", exact: true })).toBeVisible();
    expect(detailReads).toEqual(["a", "b"]);
    expect(bulkReads).toBe(0);
    await page.screenshot({ path: testInfo.outputPath("workflow-library.png"), fullPage: true });
  });
}
