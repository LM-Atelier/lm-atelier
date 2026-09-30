import { expect, test } from "@playwright/test";
import type { Run, TurnAccepted } from "../apps/web/src/types";

for (const width of [1280, 390]) {
  test(`reads and retries image history beyond the transcript page at width ${width}`, async ({ page, request }) => {
    test.setTimeout(180_000);
    await page.setViewportSize({ width, height: 844 });
    const session = await request.post("/api/session");
    const { csrf_token: csrf } = await session.json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf };
    const imported = await request.post("/api/models/import", { headers, data: {
      name: "Image history fixture", role: "chat", engine: "mock", local_path: process.env.LM_ATELIER_E2E_MODEL_PATH,
    } });
    expect(imported.status()).toBe(201);
    const created = await request.post("/api/chats", { headers,
      data: { title: `Image history ${width}`, routing_mode: "text" } });
    expect(created.status()).toBe(201);
    const { id } = await created.json() as { id: string };
    try {
      const turn = async (text: string, mode: "image" | "text", source?: string) => {
        const response = await request.post(`/api/chats/${id}/turns`, { headers, data: {
          text, mode, ...(source ? { input_artifact_ids: [source] } : {}),
        } });
        expect(response.status()).toBe(202);
        const accepted = await response.json() as TurnAccepted;
        let completed: Run | undefined;
        await expect.poll(async () => {
          const response = await request.get(`/api/runs/${accepted.run.id}`);
          expect(response.ok()).toBeTruthy();
          completed = await response.json() as Run;
          return completed.status;
        }, { intervals: [20, 50, 100] }).toBe("complete");
        return { accepted, completed: completed! };
      };
      const image = async (text: string, source?: string) => {
        const result = await turn(text, "image", source);
        const outputs = result.completed.provenance_json.outputs as { artifact_id: string }[];
        expect(outputs.length).toBeGreaterThan(0);
        return { ...result.accepted, artifactId: outputs[0].artifact_id };
      };
      const original = await image("Create an image of a blue paper boat.");
      const first = await image("Increase contrast on the paper boat.", original.artifactId);
      for (let index = 0; index < 22; index++) await turn(`Paper boat notebook ${index}.`, "text");
      const second = await image("Increase brightness on the paper boat.", first.artifactId);
      const third = await image("Increase saturation on the paper boat.", second.artifactId);
      const reads: URL[] = [];
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      page.on("request", (outgoing) => {
        const url = new URL(outgoing.url());
        if (outgoing.method() === "GET" && url.pathname.startsWith(`/api/chats/${id}`)) reads.push(url);
      });
      const lineagePath = `/api/chats/${id}/messages/${third.assistant_message.id}/lineage`;
      let failInitial = true;
      let failOlder = true;
      await page.route(`**${lineagePath}?**`, async (route) => {
        const older = new URL(route.request().url()).searchParams.has("before");
        if (older ? failOlder : failInitial) await route.fulfill({ status: 503,
          contentType: "application/json", body: JSON.stringify({ detail: "History temporarily unavailable" }) });
        else await route.continue();
      });
      await page.addInitScript((chatId) => {
        localStorage.setItem("local-lm-chat", chatId);
        sessionStorage.setItem("lm-atelier-setup-dismissed", "1");
      }, id);
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.goto("/");
      const transcript = page.locator(".messages");
      await expect(transcript.locator(":scope > article.message")).toHaveCount(40);
      await expect(transcript.getByText("Increase contrast on the paper boat.", { exact: true })).toHaveCount(0);
      const result = transcript.locator(".message.assistant").last();
      await expect(result.getByRole("alert")).toContainText("Image history could not be loaded.");
      failInitial = false;
      await result.getByRole("button", { name: "Retry image history" }).click();
      const compare = result.getByRole("button", { name: "Compare with the source", exact: true });
      await compare.click();
      const comparison = page.getByRole("dialog", { name: "Compare with the source", exact: true });
      const sourceImage = comparison.getByAltText("The source before the edit");
      await expect(sourceImage).toHaveAttribute("src", `/api/artifacts/${encodeURIComponent(second.artifactId)}/content`);
      await expect.poll(() => sourceImage.evaluate((node) => (node as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
      const editedImage = comparison.getByAltText("The edited result");
      await expect(editedImage).toHaveAttribute("src", `/api/artifacts/${encodeURIComponent(third.artifactId)}/content`);
      await expect.poll(() => editedImage.evaluate((node) => (node as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
      const slider = comparison.getByRole("slider", { name: "Comparison position" });
      await slider.focus();
      await slider.press("ArrowRight");
      await expect(slider).toHaveValue("51");
      await comparison.getByRole("button", { name: "Close comparison" }).click();
      await expect(compare).toBeFocused();

      await result.getByRole("button", { name: "Show the edit lineage" }).click();
      const history = page.getByRole("dialog", { name: "Edit lineage", exact: true });
      await expect(history.getByText("Latest 2 steps", { exact: true })).toBeVisible();
      await expect(history.locator(".lineage-steps > li")).toHaveCount(3);
      const older = history.getByRole("button", { name: "Load older edits", exact: true });
      await older.focus();
      await older.press("Enter");
      await expect(history.getByRole("alert")).toHaveText("Older edits could not be loaded.");
      await expect(history.locator(".lineage-steps > li")).toHaveCount(3);
      await expect(older).toBeFocused();
      failOlder = false;
      await older.press("Enter");
      await expect(history.locator(".lineage-steps > li")).toHaveCount(4);
      await expect(history.getByText("3 steps", { exact: true })).toBeVisible();
      await expect(history.getByText("Increase contrast on the paper boat.", { exact: true })).toBeVisible();
      await expect(history.getByAltText("What entered step 1")).toHaveAttribute("src",
        `/api/artifacts/${encodeURIComponent(original.artifactId)}/content`);
      await expect(history.getByRole("button", { name: "All edits loaded" })).toBeFocused();
      await history.getByRole("button", { name: "Close lineage" }).click();
      await expect(result.getByRole("button", { name: "Show the edit lineage" })).toBeFocused();
      await expect(transcript.locator(":scope > article.message")).toHaveCount(40);
      await expect(transcript.getByText("Increase contrast on the paper boat.", { exact: true })).toHaveCount(0);
      expect(reads.some((url) => url.pathname === `/api/chats/${id}`)).toBe(false);
      expect(reads.filter((url) => url.pathname.endsWith("/messages")).every((url) =>
        Number(url.searchParams.get("limit")) <= 40 && !url.searchParams.has("before"))).toBe(true);
      const historyReads = reads.filter((url) => url.pathname === lineagePath);
      expect(historyReads.some((url) => url.searchParams.get("limit") === "2")).toBe(true);
      expect(historyReads.some((url) => url.searchParams.get("before") === second.user_message.id
        && url.searchParams.get("limit") === "40")).toBe(true);
      expect(errors).toEqual([]);
    } finally {
      await page.goto("about:blank");
      expect.soft((await request.delete(`/api/chats/${id}`, { headers })).ok()).toBeTruthy();
    }
  });
}
