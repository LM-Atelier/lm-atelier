import { expect, test, type Page } from "@playwright/test";

const scales = { compact: 0.8, standard: 1, comfortable: 1.2 } as const;

async function visibleOverflow(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const width = window.innerWidth;
    const result: string[] = [];
    for (const element of Array.from(document.querySelectorAll<HTMLElement>("#main-content *"))) {
      const box = element.getBoundingClientRect();
      if (!box.width || !box.height || getComputedStyle(element).visibility === "hidden") continue;
      if (element.getAnimations().some(animation => animation.playState === "running")) continue;
      let parent = element.parentElement;
      let contained = false;
      while (parent && parent.id !== "main-content" && !parent.classList.contains("page-view") && parent.tagName !== "MAIN") {
        if (getComputedStyle(parent).overflowX !== "visible") {
          const bounds = parent.getBoundingClientRect();
          contained = bounds.left >= -1 && bounds.right <= width + 1;
          break;
        }
        parent = parent.parentElement;
      }
      if (!contained && (box.left < -1 || box.right > width + 1)) result.push(element.tagName + "." + element.className);
    }
    return result;
  });
}

for (const [density, scale] of Object.entries(scales)) {
  for (const width of [360, 1280]) {
    test(`${density} density fits ${width}px with larger text and keyboard controls`, async ({ page, request }, testInfo) => {
      const session = await request.post("/api/session");
      expect(session.ok()).toBeTruthy();
      const { csrf_token } = await session.json() as { csrf_token: string };
      const created = await request.post("/api/chats", {
        headers: { "x-local-lm-csrf": csrf_token }, data: { title: "Interface density" },
      });
      expect(created.status()).toBe(201);
      const { id } = await created.json() as { id: string };
      await page.setViewportSize({ width, height: 900 });
      await page.addInitScript(({ choice, chatId }) => {
        if (!localStorage.getItem("local-lm-density")) localStorage.setItem("local-lm-density", choice);
        localStorage.setItem("local-lm-text-size", "larger");
        localStorage.setItem("local-lm-chat", chatId);
      }, { choice: density, chatId: id });
      await page.goto("/?view=settings&settings=appearance");
      const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
      await expect(setup).toBeVisible();
      await setup.getByRole("button", { name: "Not now" }).click();
      const choices = page.getByRole("group", { name: "Interface density" });
      await expect(choices).toBeVisible();
      const chosen = choices.getByRole("button", { name: density, exact: false });
      await expect(chosen).toHaveAttribute("aria-pressed", "true");
      const metrics = await page.locator(".setting-row").first().evaluate(element => {
        const style = getComputedStyle(element);
        return { padding: parseFloat(style.paddingTop), gap: parseFloat(style.columnGap),
          labelSize: parseFloat(getComputedStyle(element.querySelector("strong")!).fontSize) };
      });
      expect(metrics.padding).toBeCloseTo(13 * scale, 1);
      expect(metrics.gap).toBeCloseTo(16 * scale, 1);
      expect(metrics.labelSize).toBeCloseTo(13 * 1.25, 1);
      expect(await visibleOverflow(page), "settings fits the viewport").toEqual([]);
      for (const button of await choices.getByRole("button").all()) {
        const box = await button.boundingBox();
        expect(box).not.toBeNull();
        expect(box!.width).toBeGreaterThanOrEqual(24);
        expect(box!.height).toBeGreaterThanOrEqual(24);
      }
      await choices.scrollIntoViewIfNeeded();
      await page.screenshot({ path: testInfo.outputPath("appearance.png"), fullPage: true });
      const compact = choices.getByRole("button", { name: "Compact", exact: true });
      await compact.focus();
      await page.keyboard.press("Space");
      await expect(compact).toBeFocused();
      await expect(page.locator("html")).toHaveAttribute("data-density", "compact");
      await page.reload();
      await expect(page.locator("html")).toHaveAttribute("data-density", "compact");
      await choices.getByRole("button", { name: density, exact: false }).click();
      await page.goto("/?view=chat");
      await expect(page.getByRole("textbox", { name: "Message" })).toBeVisible();
      expect(await visibleOverflow(page), "chat fits the viewport").toEqual([]);
      await page.goto("/?view=media");
      await expect(page.getByRole("heading", { name: "Media library", exact: true })).toBeVisible();
      expect(await visibleOverflow(page), "media library fits the viewport").toEqual([]);
    });
  }
}
