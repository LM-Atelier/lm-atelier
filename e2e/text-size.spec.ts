import { expect, test, type APIRequestContext, type Browser, type Page } from "@playwright/test";

/** Text size, measured in a real browser at real window sizes.
 *
 * The setting multiplies every font size in the stylesheet, and jsdom loads no
 * stylesheet, so only a browser can say whether Standard is still the size it
 * was, whether larger text really is larger, and whether it still fits a phone
 * and can still be reached from the keyboard.
 */

const TEXT_SIZE_KEY = "local-lm-text-size";
const SCALES = { standard: 1, large: 1.125, larger: 1.25 } as const;
const PHONE = { width: 360, height: 780 };
const DESKTOP = { width: 1280, height: 900 };

async function chat(request: APIRequestContext): Promise<string> {
  const session = await request.post("/api/session");
  expect(session.ok()).toBeTruthy();
  const { csrf_token } = (await session.json()) as { csrf_token: string };
  const response = await request.post("/api/chats", { headers: { "x-local-lm-csrf": csrf_token }, data: { title: "Text size" } });
  expect(response.status()).toBe(201);
  return ((await response.json()) as { id: string }).id;
}

async function open(browser: Browser, size: string, viewport: { width: number; height: number }, path: string, chatId = "") {
  const context = await browser.newContext({ viewport });
  await context.addInitScript(([key, value, current]) => {
    localStorage.setItem(key, value);
    if (current) localStorage.setItem("local-lm-chat", current);
  }, [TEXT_SIZE_KEY, size, chatId] as const);
  const page = await context.newPage();
  await page.goto(path);
  const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setup).toBeVisible();
  await setup.getByRole("button", { name: "Not now" }).click();
  await expect(setup).toBeHidden();
  return { context, page };
}

async function pixels(page: Page, selector: string): Promise<number> {
  return page.locator(selector).first().evaluate((element) => Number.parseFloat(getComputedStyle(element).fontSize));
}

/** Every visible piece of the page that runs past the side of the window.
 *
 * The workspace's main region clips its overflow, so its own scroll width
 * never grows and cannot tell a page that fits from one that does not. Each
 * element is measured against the window instead. Content inside a nested
 * container that scrolls or clips and itself fits - a wide table, a code block,
 * a moving progress bar in its track - is reachable or deliberately hidden, so
 * only that container is held to the window.
 */
async function pastTheEdge(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const width = window.innerWidth;
    const pageLevel = (node: HTMLElement) =>
      node.id === "main-content" || node.classList.contains("page-view") || node.tagName === "MAIN" || node === document.body;
    const found: string[] = [];
    for (const element of Array.from(document.querySelectorAll<HTMLElement>("#main-content *"))) {
      const rect = element.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0 || getComputedStyle(element).visibility === "hidden") continue;
      // Something in motion, like a loading bar sweeping across, is not laid out past the edge.
      if (element.getAnimations().some((animation) => animation.playState === "running")) continue;
      let scroller = element.parentElement;
      let contained = false;
      while (scroller && !pageLevel(scroller)) {
        const overflow = getComputedStyle(scroller).overflowX;
        if (overflow !== "visible") {
          const bounds = scroller.getBoundingClientRect();
          contained = bounds.right <= width + 1 && bounds.left >= -1;
          break;
        }
        scroller = scroller.parentElement;
      }
      if (contained) continue;
      if (rect.right > width + 1 || rect.left < -1) {
        found.push(`${element.tagName.toLowerCase()}.${Array.from(element.classList).join(".")} ${Math.round(rect.left)}-${Math.round(rect.right)}`);
      }
    }
    return found;
  });
}

async function mainFits(page: Page, where: string) {
  expect(await pastTheEdge(page), `${where}: nothing may spill past the side of the window`).toEqual([]);
}

for (const [size, scale] of Object.entries(SCALES)) {
  test(`${size} text is drawn at its size and keeps the hierarchy between sizes`, async ({ browser }) => {
    const measure = async (choice: string) => {
      const opened = await open(browser, choice, DESKTOP, "/?view=settings&settings=appearance");
      try {
        await expect(opened.page.getByRole("heading", { name: "Light and theme" })).toBeVisible();
        return {
          heading: await pixels(opened.page, ".detail-title h2"),
          label: await pixels(opened.page, ".setting-row strong"),
          help: await pixels(opened.page, ".setting-row small"),
          choice: await pixels(opened.page, ".segmented button"),
        };
      } finally {
        await opened.context.close();
      }
    };
    const standard = await measure("standard");
    const chosen = size === "standard" ? standard : await measure(size);
    // Standard is today's page heading exactly, and every other size is that
    // same layout multiplied, so no piece of text changes rank.
    expect(standard.heading).toBe(20);
    for (const key of Object.keys(standard) as (keyof typeof standard)[]) {
      expect(chosen[key], key).toBeCloseTo(standard[key] * scale, 1);
    }
    expect(chosen.heading).toBeGreaterThan(chosen.label);
  });

  test(`${size} text fits a phone and stays reachable from the keyboard`, async ({ browser, request }) => {
    const phone = await open(browser, size, PHONE, "/?view=chat", await chat(request));
    try {
      await mainFits(phone.page, "chat");
      await phone.page.getByRole("link", { name: "Skip to main content" }).focus();
      await phone.page.keyboard.press("Enter");
      await expect(phone.page.locator("#main-content")).toBeFocused();
      const composer = phone.page.getByRole("textbox", { name: "Message" });
      await composer.focus();
      await expect(composer).toBeFocused();
      const box = await composer.boundingBox();
      expect(box, "the composer is laid out").not.toBeNull();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(PHONE.width + 0.5);
      await phone.page.keyboard.press("Tab");
      const focusedInside = await phone.page.evaluate(() => {
        const active = document.activeElement as HTMLElement | null;
        if (!active || active === document.body) return false;
        const rect = active.getBoundingClientRect();
        return rect.right <= window.innerWidth + 0.5 && rect.left >= -0.5;
      });
      expect(focusedInside, "the next control after the composer is on screen").toBe(true);

      await phone.page.goto("/?view=settings&settings=appearance");
      await expect(phone.page.getByRole("heading", { name: "Light and theme" })).toBeVisible();
      await mainFits(phone.page, "settings");
      await phone.page.goto("/?view=media");
      await mainFits(phone.page, "media library");
    } finally {
      await phone.context.close();
    }
  });
}
