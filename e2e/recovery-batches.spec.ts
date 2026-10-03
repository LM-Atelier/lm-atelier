import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import type { RecoveryBatchResult, RecoveryImpact, RecoveryItem } from "../apps/web/src/recoveryTypes";
import type { Workflow, WorkflowFamily } from "../apps/web/src/types";

type AccessibilityNode = {
  nodeId: string; childIds?: string[]; role?: { value?: string }; name?: { value?: string };
  properties?: { name: string; value: { value?: unknown } }[];
};

async function checkAccessibleConfirmation(page: Page, title: string, labels: string[]) {
  const surface = page.getByRole("dialog", { name: title, exact: true });
  for (const label of labels) await expect(surface.getByText(label, { exact: true })).toBeVisible();
  const session = await page.context().newCDPSession(page);
  try {
    const { nodes } = await session.send("Accessibility.getFullAXTree") as { nodes: AccessibilityNode[] };
    const dialog = nodes.find(node => node.role?.value === "dialog" && node.name?.value === title);
    expect(dialog).toBeDefined();
    expect(dialog!.properties?.find(property => property.name === "modal")?.value.value).toBe(true);
    const byId = new Map(nodes.map(node => [node.nodeId, node]));
    const text: string[] = [];
    const visit = (node: AccessibilityNode) => {
      if (node.role?.value === "StaticText" && node.name?.value) text.push(node.name.value);
      for (const id of node.childIds ?? []) {
        const child = byId.get(id);
        if (child) visit(child);
      }
    };
    visit(dialog!);
    for (const label of labels) expect(text).toContain(label);
    expect(text.some(value => value.startsWith("All or nothing:"))).toBe(true);
  } finally {
    await session.detach();
  }
}

async function trash(request: APIRequestContext, headers: Record<string, string>, path: string, key: string) {
  const response = await request.get(`${path}/deletion-impact`);
  expect(response.status()).toBe(200);
  const preview = await response.json() as RecoveryImpact;
  const moved = await request.post(`${path}/trash`, { headers, data: {
    expected_revision: preview.revision, impact_sha256: preview.impact_sha256, operation_key: key,
  } });
  expect(moved.status()).toBe(200);
  return await moved.json() as RecoveryItem;
}

for (const [width, zoom] of [[1280, 1], [390, 1], [1280, 2]]) {
  test(`restores and permanently deletes four kinds together at ${width}px and ${zoom * 100}% zoom`, async ({ page, request }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width, height: 900 });
    if (zoom === 2) await page.addInitScript(() => {
      document.addEventListener("DOMContentLoaded", () => { document.documentElement.style.zoom = "2"; });
    });
    await page.route("**/api/setup/readiness", route => route.fulfill({ json: { version: 2, state: "ready", roles: [] } }));
    await page.addInitScript(() => { sessionStorage.setItem("lm-atelier-setup-dismissed", "1"); });
    const { csrf_token } = await (await request.post("/api/session")).json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf_token };
    const suffix = `${width}-${zoom}`;
    const createdProject = await request.post("/api/projects", { headers, data: { name: `Garden project ${suffix}` } });
    expect(createdProject.status()).toBe(201);
    const project = await createdProject.json() as { id: string };
    const createdChat = await request.post("/api/chats", { headers, data: { title: `Garden conversation ${suffix}`, project_id: project.id, routing_mode: "text" } });
    expect(createdChat.status()).toBe(201);
    const chat = await createdChat.json() as { id: string };
    const imported = await request.post("/api/models/import", { headers, data: {
      name: `Garden batch chat ${suffix}`, role: "chat", engine: "mock", local_path: process.env.LM_ATELIER_E2E_MODEL_PATH,
    } });
    expect(imported.status()).toBe(201);
    const install = await imported.json() as { id: string };
    const profiles = await (await request.get("/api/profiles")).json() as { id: string; model_install_id: string | null }[];
    const profile = profiles.find(item => item.model_install_id === install.id);
    if (!profile) throw new Error("Imported fixture has no chat profile.");
    expect((await request.patch(`/api/chats/${chat.id}`, { headers, data: { active_chat_profile_id: profile.id } })).status()).toBe(200);
    const accepted = await request.post(`/api/chats/${chat.id}/turns`, { headers, data: { text: "Keep the garden paths wide", mode: "text", profile_id: profile.id } });
    expect(accepted.status()).toBe(202);
    const run = await accepted.json() as { run: { id: string } };
    await expect.poll(async () => (await (await request.get(`/api/runs/${run.run.id}`)).json()).status).toBe("complete");
    const history = await (await request.get(`/api/chats/${chat.id}/messages`)).json();
    expect(history.messages).toHaveLength(2);
    const uploaded = await request.post("/api/artifacts?kind=image", { headers, multipart: {
      file: { name: `Garden image ${suffix}.png`, mimeType: "image/png", buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6V8AAAAASUVORK5CYII=", "base64") },
    } });
    expect(uploaded.status()).toBe(201);
    const artifact = await uploaded.json() as { id: string };
    const artifactPath = `/api/artifacts/${encodeURIComponent(artifact.id)}`;
    expect((await request.patch(artifactPath, { headers, data: { favorite: true } })).status()).toBe(200);
    const library = await (await request.get("/api/artifact-library")).json() as { items: { id: string; artifact_id: string }[] };
    const entry = library.items.find(item => item.artifact_id === artifact.id);
    if (!entry) throw new Error("Uploaded fixture has no library membership.");
    const createdWorkflow = await request.post("/api/workflows", { headers, data: {
      name: `Garden workflow ${suffix}`, operation: "text_to_image", engine: "mock",
      api_graph: { garden: { class_type: "NeutralFixture", inputs: { spacing: 37 } } },
    } });
    expect(createdWorkflow.status()).toBe(201);
    const workflow = await createdWorkflow.json() as Workflow;
    const originalWorkflow = await (await request.get(`/api/workflows/${workflow.id}`)).json() as Workflow;
    const paths = [`/api/projects/${project.id}`, `/api/chats/${chat.id}`, `/api/artifact-library/${encodeURIComponent(entry.id)}`, `/api/workflow-families/${workflow.family_id}`];
    let items: RecoveryItem[] = [];
    for (const [index, path] of paths.entries()) items.push(await trash(request, headers, path, `trash-batch-${suffix}-${index}`));
    const requests: { path: string; body: Record<string, unknown> }[] = [];
    page.on("request", value => {
      if (value.method() === "POST" && new URL(value.url()).pathname.startsWith("/api/recovery-items/"))
        requests.push({ path: new URL(value.url()).pathname, body: value.postDataJSON() as Record<string, unknown> });
    });
    await page.goto("/?view=settings&settings=data-and-backups");
    const center = page.locator("section", { has: page.getByRole("heading", { name: "Recently Deleted", exact: true }) });
    const select = async () => {
      for (const item of items) {
        const checkbox = center.getByRole("checkbox", { name: `Select ${item.display_label}`, exact: true });
        await checkbox.focus(); await checkbox.press("Space"); await expect(checkbox).toBeChecked();
      }
      await expect(center.getByText("4 of 20 items selected", { exact: true })).toBeVisible();
    };
    await select();
    const reviewRestore = center.getByRole("button", { name: "Review restore selection", exact: true });
    await reviewRestore.focus(); await reviewRestore.press("Enter");
    const restoreDialog = page.getByRole("dialog", { name: "Restore selected items?", exact: true });
    const restore = restoreDialog.getByRole("button", { name: "Restore selected items", exact: true });
    await expect(restore).toBeEnabled();
    await expect(restoreDialog.getByText(/All or nothing/)).toBeVisible();
    await expect(restoreDialog.getByText(/does not grant trust or activate it/)).toBeVisible();
    await checkAccessibleConfirmation(page, "Restore selected items?", items.map(item => item.display_label));
    const closeRestore = restoreDialog.getByRole("button", { name: "Close without changing anything", exact: true });
    await expect(closeRestore).toBeFocused();
    await closeRestore.press("Shift+Tab"); await expect(restore).toBeFocused();
    await restore.press("Tab"); await expect(closeRestore).toBeFocused();
    for (const item of items) await expect(restoreDialog.locator(`time[datetime="${item.purge_after}"]`).first()).toBeAttached();
    const previewResponse = requests.filter(value => value.path === "/api/recovery-items/batches");
    expect(previewResponse).toHaveLength(1);
    expect(previewResponse[0].body).toEqual({ deletion_ids: items.map(item => item.deletion_id), action: "restore", restore_unfiled: false });
    const applied = page.waitForResponse(response => /\/api\/recovery-items\/batches\/[^/]+\/apply$/.test(new URL(response.url()).pathname));
    await restore.focus(); await restore.press("Enter");
    const restoredResponse = await applied; expect(restoredResponse.status()).toBe(200);
    const restoredResult = await restoredResponse.json() as RecoveryBatchResult;
    expect(restoredResult).toMatchObject({ action: "restore", policy: "all-or-nothing", reclaimed_bytes: 0 });
    expect(restoredResult.results).toHaveLength(4);
    const notice = center.getByText(/4 items restored together/);
    await expect(notice).toBeFocused(); await expect(restoreDialog).toHaveCount(0);
    expect(await (await request.get(`/api/chats/${chat.id}/messages`)).json()).toEqual(history);
    expect((await (await request.get(`/api/chats/${chat.id}/metadata`)).json()).project_id).toBe(project.id);
    const restoredLibrary = await (await request.get("/api/artifact-library")).json() as { items: { id: string; favorite: boolean }[] };
    expect(restoredLibrary.items.find(item => item.id === entry.id)?.favorite).toBe(true);
    const restoredWorkflow = await (await request.get(`/api/workflows/${workflow.id}`)).json() as Workflow;
    expect(restoredWorkflow.revisions).toEqual(originalWorkflow.revisions);
    const family = await (await request.get(paths[3])).json() as WorkflowFamily;
    expect(family.enabled).toBe(false); expect(family.preferences.every(value => !value.enabled && !value.is_default)).toBe(true);
    expect((await request.get(`${artifactPath}/content`)).status()).toBe(200);
    const command = requests.find(value => value.path.endsWith("/apply"))!;
    expect((await request.post(command.path, { headers, data: command.body })).status()).toBe(200);
    expect(await (await request.post(command.path, { headers, data: command.body })).json()).toEqual(restoredResult);

    items = [];
    for (const [index, path] of paths.entries()) items.push(await trash(request, headers, path, `trash-again-batch-${suffix}-${index}`));
    await center.getByRole("button", { name: "Refresh", exact: true }).click();
    await select();
    await center.getByRole("button", { name: "Review permanent deletion", exact: true }).click();
    const purgeDialog = page.getByRole("dialog", { name: "Permanently delete selected items?", exact: true });
    const purge = purgeDialog.getByRole("button", { name: "Delete selected permanently", exact: true });
    await expect(purgeDialog.getByText(/Per-item storage estimates may overlap/)).toBeVisible();
    await checkAccessibleConfirmation(page, "Permanently delete selected items?", items.map(item => item.display_label));
    await expect(purge).toBeDisabled();
    await purgeDialog.getByRole("button", { name: "Cancel", exact: true }).click();
    expect(requests.filter(value => value.path.endsWith("/apply"))).toHaveLength(1);
    await center.getByRole("button", { name: "Review permanent deletion", exact: true }).click();
    await expect(purge).toBeDisabled();
    const acknowledgement = purgeDialog.getByRole("checkbox", { name: /permanent deletion cannot be undone/ });
    await acknowledgement.focus(); await acknowledgement.press("Space");
    await expect(purge).toBeEnabled();
    const purged = page.waitForResponse(response => /\/api\/recovery-items\/batches\/[^/]+\/apply$/.test(new URL(response.url()).pathname));
    await purge.focus(); await purge.press("Enter");
    const response = await purged; expect(response.status()).toBe(200);
    const purgedResult = await response.json() as RecoveryBatchResult;
    expect(purgedResult).toMatchObject({ action: "purge", policy: "all-or-nothing", reclaimed_bytes: 0 });
    expect(purgedResult.results).toHaveLength(4);
    expect(purgedResult.results.map(item => item.deletion_id).sort()).toEqual(items.map(item => item.deletion_id).sort());
    expect(purgedResult.results.every(item => item.action === "purge")).toBe(true);
    await expect(center.getByText(/4 items permanently deleted together/)).toBeFocused();
    const pageItems = await (await request.get("/api/recovery-items?state=purged")).json() as { items: RecoveryItem[] };
    for (const item of items) expect(pageItems.items.some(value => value.deletion_id === item.deletion_id)).toBe(false);
    expect((await request.get(`/api/chats/${chat.id}/metadata`)).status()).toBe(404);
    expect((await request.get(`/api/workflows/${workflow.id}`)).status()).toBe(404);
    const projects = await (await request.get("/api/projects")).json() as { id: string }[];
    expect(projects.some(item => item.id === project.id)).toBe(false);
    const remainingLibrary = await (await request.get("/api/artifact-library")).json() as { items: { id: string }[] };
    expect(remainingLibrary.items.some(item => item.id === entry.id)).toBe(false);
    expect((await request.get(`${artifactPath}/content`)).status()).toBe(200);
    expect(requests.filter(value => value.path.endsWith("/apply"))).toHaveLength(2);
    expect(requests.every(value => value.path.startsWith("/api/recovery-items/batches"))).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)).toBe(false);
  });
}
