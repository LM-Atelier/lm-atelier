import { expect, test } from "@playwright/test";

test("pages the workspace while preserving saved selection and searching unloaded chats", async ({ page, request }) => {
  const session = await request.post("/api/session");
  const { csrf_token: token } = await session.json() as { csrf_token: string };
  const headers = { "x-local-lm-csrf": token };
  const chatIds: string[] = [];
  let projectId: string | null = null;
  try {
    const projectResponse = await request.post("/api/projects", { headers, data: { name: "Harbor project" } });
    expect(projectResponse.status()).toBe(201);
    const project = await projectResponse.json() as { id: string };
    projectId = project.id;
    const oldestResponse = await request.post("/api/chats", { headers, data: { title: "Old sketch", project_id: project.id } });
    expect(oldestResponse.status()).toBe(201);
    const oldest = await oldestResponse.json() as { id: string };
    chatIds.push(oldest.id);
    for (let number = 0; number < 55; number++) {
      const response = await request.post("/api/chats", { headers, data: { title: `Notebook ${number}` } });
      expect(response.status()).toBe(201);
      const chat = await response.json() as { id: string };
      chatIds.push(chat.id);
    }
    const archivedResponse = await request.post("/api/chats", { headers, data: { title: "Archived sketch" } });
    expect(archivedResponse.status()).toBe(201);
    const archived = await archivedResponse.json() as { id: string };
    chatIds.push(archived.id);
    expect((await request.patch(`/api/chats/${archived.id}`, { headers, data: { archived: true } })).status()).toBe(200);

    await page.addInitScript((id: string) => {
      localStorage.setItem("local-lm-chat", id);
      sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
    }, oldest.id);
    const requests: URL[] = [];
    page.on("request", (message) => {
      const url = new URL(message.url());
      if (url.pathname === "/api/chats" && message.method() === "GET") requests.push(url);
    });
    await page.goto("/");
    const workspace = page.getByRole("region", { name: "Projects and chats" });
    await expect(page.getByRole("heading", { name: "Old sketch", exact: true })).toBeVisible();
    await expect(workspace.locator(".sidebar-chat-row")).toHaveCount(50);
    await expect(workspace.getByRole("button", { name: "Old sketch", exact: true })).toHaveCount(0);

    await page.getByLabel("Search projects and chats").fill("Harbor");
    await expect(workspace.getByRole("button", { name: "Old sketch", exact: true })).toBeVisible();
    await expect(workspace.locator(".sidebar-chat-row")).toHaveCount(1);
    await page.getByLabel("Search projects and chats").fill("");
    await workspace.getByRole("button", { name: "Load more chats" }).click();
    await expect(workspace.locator(".sidebar-chat-row")).toHaveCount(56);
    await workspace.getByRole("button", { name: "Notebook 0", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Notebook 0", exact: true })).toBeVisible();

    await page.getByLabel("Search projects and chats").fill("Archived sketch");
    await expect(workspace.locator(".sidebar-chat-row")).toHaveCount(0);
    await page.getByRole("button", { name: "Archived", exact: true }).click();
    await expect(workspace.getByRole("button", { name: "Archived sketch Archived", exact: true })).toBeVisible();
    expect(requests.length).toBeGreaterThan(3);
    expect(requests.every((url) => url.searchParams.get("limit") === "50")).toBe(true);
    expect(requests.some((url) => url.searchParams.get("offset") === "50")).toBe(true);
  } finally {
    await page.goto("about:blank");
    for (const id of chatIds) {
      const response = await request.delete(`/api/chats/${id}`, { headers });
      expect.soft(response.ok(), `remove fixture chat ${id}`).toBeTruthy();
    }
    if (projectId !== null) {
      const response = await request.delete(`/api/projects/${projectId}`, { headers });
      expect.soft(response.ok(), "remove fixture project").toBeTruthy();
    }
  }
});
