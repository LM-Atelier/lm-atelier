import { expect, test } from "@playwright/test";

for (const width of [1280, 375]) {
  test.describe(`generation retries at ${width} pixels`, () => {
    test.use({ viewport: { width, height: 800 } });
    test("saves the retry allowance by keyboard and keeps it after reload", async ({ page }) => {
      await page.goto("/?view=settings&settings=general");
      await page.getByRole("dialog", { name: "Set up LM Atelier" })
        .getByRole("button", { name: "Not now" }).click();
      const input = page.getByRole("spinbutton", { name: "Automatic retries" });
      await expect(input).toBeVisible();
      const original = await input.inputValue();
      const choice = original === "0" ? "3" : "0";
      await input.fill(choice);
      await input.press("Tab");
      const save = page.getByRole("button", { name: "Save retries" });
      await expect(save).toBeFocused();
      await page.keyboard.press("Enter");
      await expect(page.getByText("Automatic retry setting saved.")).toBeVisible();
      await expect(save).toBeFocused();
      expect(await page.locator(".settings-destination").evaluate(
        (element) => element.scrollWidth <= element.clientWidth + 1,
      )).toBe(true);
      await page.reload();
      await expect(input).toHaveValue(choice);
      await input.fill(original);
      await save.click();
      await expect(page.getByText("Automatic retry setting saved.")).toBeVisible();
    });
  });
}
