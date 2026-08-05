import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

/** The studio's honesty about what it cannot do, driven through the browser.
 *
 * The project had exactly one end-to-end test, and the studio is where three
 * separate defects reached the owner: a masked edit refused by the settings
 * hierarchy, a selection that only appeared after the stroke ended, and every
 * tool looking ready whether or not any installed workflow could honor it.
 *
 * Unit tests cover each of those now. What no test covered is the surface as a
 * person meets it: open the studio with nothing installed and see whether it
 * says so before the work or after it.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

async function createSession(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/session");
  expect(response.ok()).toBeTruthy();
  const payload = (await response.json()) as { csrf_token: string };
  return payload.csrf_token;
}

test("says which studio tools cannot run before any work is done", async ({ page, request }) => {
  const csrfToken = await createSession(request);
  const capabilities = await request.get("/api/studio/capabilities", {
    headers: { "x-local-lm-csrf": csrfToken },
  });
  expect(capabilities.ok()).toBeTruthy();
  const report = (await capabilities.json()) as {
    tools: { kind: string; available: boolean; reason: string | null }[];
  };

  // With nothing installed every tool is unavailable, and each says what would
  // fix it. A tool that is unavailable and silent is the defect this replaced.
  const blocked = report.tools.filter((tool) => !tool.available);
  expect(blocked.length).toBeGreaterThan(0);
  for (const tool of blocked) {
    expect(tool.reason, `${tool.kind} must say what is missing`).toBeTruthy();
  }

  await page.goto("/");
  await dismissSetup(page);
  // The nav entry specifically: a chat image also offers "Open in Image
  // Studio", and matching both is how a test starts depending on which one
  // renders first.
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();

  // With no picture open the studio asks for one rather than showing a rail of
  // tools with nothing to point at.
  await expect(page.getByRole("heading", { name: "Open an image to edit" })).toBeVisible();
  // Attached rather than visible: the file input is deliberately hidden
  // behind a styled control, which is how every custom picker works.
  await expect(page.getByLabel("Choose an image to edit")).toBeAttached();
});

/** Not covered here, and worth saying rather than leaving as an apparent gap:
 * the guided tool rail itself. Reaching it needs a picture open in the studio,
 * which means either generating one through the golden path or crossing from
 * the media library. Both are real journeys and both belong in a test; neither
 * is this one, and a test whose name promised the rail while asserting the
 * empty state would be worse than none. The rail's behaviour is covered by
 * StudioToolRail's own tests and the capability report above. */
