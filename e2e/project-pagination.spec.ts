import { expect, test } from "@playwright/test";

for (const width of [1280, 375]) {
  test(`page projects and refile a chat outside the first page at ${width}px`, async ({ page, request }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width, height: 812 });
    const { csrf_token } = await (await request.post("/api/session")).json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf_token };
    const projects: Array<{ id: string; name: string }> = [];
    let chatId: string | null = null;
    const deletedProjects = new Set<string>();
    const prefix = `Paged ${width} notebook`;
    try {
      for (let number = 0; number < 56; number++) {
        const response = await request.post("/api/projects", { headers, data: { name: `${prefix} ${number}` } });
        expect(response.status()).toBe(201);
        projects.push(await response.json() as { id: string; name: string });
      }
      const response = await request.post("/api/chats", {
        headers, data: { title: `Distant study ${width}`, project_id: projects[0].id },
      });
      expect(response.status()).toBe(201);
      chatId = (await response.json() as { id: string }).id;
      const firstPage = await (await request.get(`/api/projects?limit=50&query=${encodeURIComponent(prefix)}`)).json() as Array<{ id: string }>;
      expect(firstPage).toHaveLength(50);
      expect(firstPage.some((project) => project.id === projects[0].id)).toBe(false);
      await page.addInitScript((id: string) => {
        localStorage.setItem("local-lm-chat", id);
        sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
      }, chatId);
      const reads: URL[] = [];
      page.on("request", (message) => {
        const url = new URL(message.url());
        if (url.pathname === "/api/projects" && message.method() === "GET") reads.push(url);
      });
      await page.goto("/");
      await expect(page.getByRole("region", { name: `Distant study ${width}`, exact: true })).toBeVisible();
      if (width === 375) await page.getByRole("button", { name: "Toggle navigation" }).click();
      const workspace = page.getByRole("region", { name: "Projects and chats" });
      await page.getByLabel("Search projects and chats").fill(prefix);
      await expect(workspace.getByRole("button", { name: `Distant study ${width}`, exact: true })).toBeVisible();
      await expect(workspace.locator(".project-group")).toHaveCount(51);
      await workspace.getByRole("button", { name: "Load more projects" }).click();
      await expect(workspace.locator(".project-group")).toHaveCount(56);
      await workspace.getByRole("button", { name: `Manage Distant study ${width}` }).click();
      const manager = page.getByRole("dialog", { name: "Chat settings" });
      await expect(manager.getByRole("combobox", { name: "Project", exact: true })).toHaveValue(projects[0].id);
      await manager.getByRole("searchbox", { name: "Search projects", exact: true }).fill(`${prefix} 7`);
      await manager.getByRole("combobox", { name: "Project", exact: true }).selectOption(projects[7].id);
      expect(await manager.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
      await manager.getByRole("button", { name: "Save chat" }).click();
      await expect.poll(async () => (await (await request.get(`/api/chats/${chatId}`)).json() as { project_id: string }).project_id).toBe(projects[7].id);
      // Deleting the selected project keeps the open conversation usable.
      await workspace.getByRole("button", { name: `Manage ${projects[7].name}`, exact: true }).click();
      await page.getByRole("dialog", { name: "Manage project", exact: true })
        .getByRole("button", { name: "Delete project", exact: true }).click();
      const unfiledMetadata = page.waitForResponse((result) =>
        new URL(result.url()).pathname === `/api/chats/${chatId}/metadata` && result.ok());
      const deleted = page.waitForResponse((result) =>
        new URL(result.url()).pathname === `/api/projects/${projects[7].id}`
          && result.request().method() === "DELETE" && result.ok());
      await page.getByRole("dialog", { name: `Delete ${projects[7].name}?`, exact: true })
        .getByRole("button", { name: "Delete project", exact: true }).click();
      await deleted;
      deletedProjects.add(projects[7].id);
      expect((await (await unfiledMetadata).json() as { project_id: string | null }).project_id).toBeNull();
      if (width === 375) await page.getByRole("button", { name: "Toggle navigation" }).click();
      await expect(page.getByRole("region", { name: `Distant study ${width}`, exact: true })).toBeVisible();
      await expect(page.getByText("Could not load project settings.", { exact: false })).toHaveCount(0);
      expect(reads.length).toBeGreaterThan(2);
      expect(reads.every((url) => ["50", "200"].includes(url.searchParams.get("limit") ?? ""))).toBe(true);
      expect(reads.some((url) => url.searchParams.get("offset") === "50")).toBe(true);
      expect(reads.some((url) => url.searchParams.getAll("project_id").includes(projects[0].id))).toBe(true);
    } finally {
      await page.goto("about:blank");
      if (chatId) expect.soft((await request.delete(`/api/chats/${chatId}`, { headers })).ok()).toBe(true);
      for (const project of projects.filter((item) => !deletedProjects.has(item.id))) {
        expect.soft((await request.delete(`/api/projects/${project.id}`, { headers })).ok()).toBe(true);
      }
    }
  });
}
