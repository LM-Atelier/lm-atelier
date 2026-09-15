import { expect, test } from "@playwright/test";
import type { Workflow, WorkflowActivationPreparation, WorkflowActivationRequest } from "../apps/web/src/types";

const contract = "b".repeat(64);
const artifact = "a".repeat(64);
const stamp = "2026-09-01T00:00:00Z";
const workflow: Workflow = {
  id: "activation-example", family_id: null, name: "Example activation", description: "Neutral workflow",
  operation: "text_to_image", current_revision_id: "example-r1",
  revisions: [{
    id: "example-r1", workflow_id: "activation-example", version: 1, engine: "mock",
    engine_version: null, trusted: true, created_at: stamp, dependency_contract_sha256: contract,
    api_graph_json: {}, ui_graph_json: {}, input_schema_json: {},
    dependencies_json: { version: 1, slots: [] },
  }],
};
const preparation: WorkflowActivationPreparation = {
  workflow_revision_id: "example-r1", workflow_artifact_sha256: artifact,
  dependency_contract_sha256: contract, state: "needs_attention", selections: null,
  slots: [{
    name: "image_style", resource_kind: "model_asset", required: true, satisfaction: "all_of",
    requirement_keys: ["default"], choices: ["Soft light", "Detailed texture"].map((name, index) => ({
      name, selection: {
        slot_name: "image_style", requirement_key: "default", local_kind: "model_asset",
        local_id: "asset-" + index, recorded_resource_identity_sha256: "c".repeat(64), mount: {},
      },
    })),
  }],
  issues: [{ code: "ambiguous_dependency_binding", slot_name: "image_style" }],
};

for (const width of [1280, 375]) {
  test("workflow activation requires the named dependency choice at " + width + "px", async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    let preparations = 0;
    const submissions: WorkflowActivationRequest[] = [];
    await page.route("**/api/workflow-families?*", route => route.fulfill({ json: [] }));
    await page.route("**/api/workflow-summaries", route => route.fulfill({ json: [{
      id: workflow.id, family_id: null, name: workflow.name, description: workflow.description,
      operation: workflow.operation, current_revision_id: workflow.current_revision_id,
      revision_count: 1, created_at: stamp, updated_at: stamp,
    }] }));
    await page.route("**/api/workflows/activation-example", route => route.fulfill({ json: workflow }));
    await page.route("**/api/workflows/activation-example/revisions/example-r1/activation/prepare", route => {
      preparations += 1;
      return route.fulfill({ json: preparation });
    });
    await page.route("**/api/workflows/activation-example/revisions/example-r1/activation", route => {
      expect(route.request().method()).toBe("POST");
      submissions.push(route.request().postDataJSON() as WorkflowActivationRequest);
      return route.fulfill({ json: {
        id: "example-activation", workflow_revision_id: "example-r1", dependency_contract_sha256: contract,
        binding_sha256: "d".repeat(64), launch_sha256: "e".repeat(64), state: "ready", is_active: true,
      } });
    });
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    if (width < 700) await page.getByRole("button", { name: "Toggle navigation" }).click();
    await page.getByRole("button", { name: "Workflows", exact: true }).click();
    await page.getByRole("button", { name: /Example activation/ }).click();
    const panel = page.getByRole("region", { name: "Workflow dependencies" });
    await expect(panel).toBeVisible();
    expect(preparations).toBe(0);
    const activate = panel.getByRole("button", { name: "Activate dependencies" });
    await activate.click();
    await expect(activate).toBeDisabled();
    expect(submissions).toEqual([]);
    const choice = panel.getByRole("combobox", { name: "image style" });
    await choice.selectOption({ label: "Detailed texture" });
    await expect(activate).toBeEnabled();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await testInfo.attach("workflow-activation-choices", {
      body: await page.screenshot({ fullPage: true }), contentType: "image/png",
    });
    await activate.click();
    await expect(panel.getByRole("status")).toHaveText("Dependencies activated.");
    expect(preparations).toBe(2);
    expect(submissions).toEqual([{
      workflow_artifact_sha256: artifact, dependency_contract_sha256: contract,
      selections: [preparation.slots[0].choices[1].selection],
    }]);
    await expect(panel).not.toContainText(contract);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  });
}
