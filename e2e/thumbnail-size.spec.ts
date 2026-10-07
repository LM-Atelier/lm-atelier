import { expect, test, type APIRequestContext, type Browser, type Page } from "@playwright/test";

/** Thumbnail sizes, measured in a real browser at a real window size.
 *
 * The setting only changes lengths in the stylesheet, and jsdom loads no
 * stylesheet and lays nothing out, so whether a large card still fits a phone
 * can only be answered here. Each size is measured twice: on a narrow window,
 * where nothing may spill past the edge, and on a wide one, where the chosen
 * size must actually be the size drawn.
 */

const THUMBNAIL_KEY = "local-lm-thumbnails";

/** A plain 96 by 64 picture in one colour, generated for this test. */
const PICTURE = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAAAaUlEQVR42u3QMQ0AAAgDsOlECUrQiwNujiZV0FQPhygQJEiQIEGCBAlCkCBBggQJEiQIQYIECRIkSJAgBAkSJEiQIEGCBCFIkCBBggQJEoQgQYIECRIkSBCCBAkSJEiQIEGCECRIkKB/FhxmwfCrGQhCAAAAAElFTkSuQmCC",
  "base64",
);

const SIZES = {
  small: { cardMin: 180, image: 150, attachment: { width: 44, height: 36 } },
  medium: { cardMin: 250, image: 210, attachment: { width: 58, height: 48 } },
  large: { cardMin: 340, image: 290, attachment: { width: 88, height: 72 } },
} as const;

const PHONE = { width: 360, height: 780 };
const DESKTOP = { width: 1280, height: 900 };

async function csrf(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/session");
  expect(response.ok()).toBeTruthy();
  return ((await response.json()) as { csrf_token: string }).csrf_token;
}

async function libraryPicture(request: APIRequestContext) {
  const response = await request.post("/api/artifacts?kind=image", {
    headers: { "x-local-lm-csrf": await csrf(request) },
    multipart: { file: { name: "plain.png", mimeType: "image/png", buffer: PICTURE } },
  });
  expect(response.status()).toBe(201);
}

async function chat(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/chats", {
    headers: { "x-local-lm-csrf": await csrf(request) },
    data: { title: "Thumbnails" },
  });
  expect(response.status()).toBe(201);
  return ((await response.json()) as { id: string }).id;
}

async function open(browser: Browser, size: string, viewport: { width: number; height: number }, path: string, chatId?: string) {
  const context = await browser.newContext({ viewport });
  await context.addInitScript(([key, value, current]) => {
    localStorage.setItem(key, value);
    if (current) localStorage.setItem("local-lm-chat", current);
  }, [THUMBNAIL_KEY, size, chatId ?? ""] as const);
  const page = await context.newPage();
  await page.goto(path);
  const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setup).toBeVisible();
  await setup.getByRole("button", { name: "Not now" }).click();
  await expect(setup).toBeHidden();
  return { context, page };
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

async function mainFits(page: Page) {
  expect(await pastTheEdge(page), "nothing may spill past the side of the window").toEqual([]);
}

for (const [size, expected] of Object.entries(SIZES)) {
  test(`${size} Media Library thumbnails fit a phone and take their size on a desktop`, async ({ browser, request }) => {
    await libraryPicture(request);

    const phone = await open(browser, size, PHONE, "/?view=media");
    try {
      await expect(phone.page.locator(".gallery-card > img").first()).toBeVisible();
      const grid = await phone.page.locator(".media-grid").evaluate((element) => ({
        scrollWidth: element.scrollWidth,
        clientWidth: element.clientWidth,
        right: element.getBoundingClientRect().right,
        cardRights: Array.from(element.querySelectorAll(".gallery-card"), (card) => card.getBoundingClientRect().right),
      }));
      expect(grid.scrollWidth).toBeLessThanOrEqual(grid.clientWidth);
      for (const right of grid.cardRights) expect(right).toBeLessThanOrEqual(grid.right + 0.5);
      await mainFits(phone.page);
    } finally {
      await phone.context.close();
    }

    const desktop = await open(browser, size, DESKTOP, "/?view=media");
    try {
      await expect(desktop.page.locator(".gallery-card > img").first()).toBeVisible();
      const drawn = await desktop.page.locator(".media-grid").evaluate((element) => ({
        firstColumn: Number.parseFloat(getComputedStyle(element).gridTemplateColumns.split(" ")[0]),
        imageHeight: element.querySelector(".gallery-card > img")!.getBoundingClientRect().height,
      }));
      expect(drawn.imageHeight).toBeCloseTo(expected.image, 0);
      expect(drawn.firstColumn).toBeGreaterThanOrEqual(expected.cardMin);
      expect(drawn.firstColumn).toBeLessThan(expected.cardMin * 2);
    } finally {
      await desktop.context.close();
    }
  });

  test(`${size} attached pictures take their size and fit a phone`, async ({ browser, request }) => {
    const chatId = await chat(request);
    const phone = await open(browser, size, PHONE, "/?view=chat", chatId);
    try {
      const chooser = phone.page.waitForEvent("filechooser");
      await phone.page.getByRole("button", { name: "Attach file" }).click();
      await (await chooser).setFiles({ name: "plain.png", mimeType: "image/png", buffer: PICTURE });
      const preview = phone.page.locator(".attachment-preview").first();
      await expect(preview).toBeVisible();
      const box = await preview.evaluate((element) => {
        const rect = element.getBoundingClientRect();
        const strip = element.closest(".attachment-strip")!.getBoundingClientRect();
        const card = element.closest(".attachment-card")!.getBoundingClientRect();
        return { width: rect.width, height: rect.height, cardRight: card.right, stripRight: strip.right };
      });
      expect(box.width).toBeCloseTo(expected.attachment.width, 0);
      expect(box.height).toBeCloseTo(expected.attachment.height, 0);
      expect(box.cardRight).toBeLessThanOrEqual(box.stripRight + 0.5);
      await mainFits(phone.page);
    } finally {
      await phone.context.close();
    }
  });
}
