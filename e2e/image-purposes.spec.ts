import { expect, test } from "@playwright/test";

const PICTURE = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAAAaUlEQVR42u3QMQ0AAAgDsOlECUrQiwNujiZV0FQPhygQJEiQIEGCBAlCkCBBggQJEiQIQYIECRIkSJAgBAkSJEiQIEGCBCFIkCBBggQJEoQgQYIECRIkSBCCBAkSJEiQIEGCECRIkKB/FhxmwfCrGQhCAAAAAElFTkSuQmCC",
  "base64",
);

for (const width of [360, 1280]) {
  test(`keeps each picture's purpose in its draft without spilling at ${width}px`, async ({ page, request }) => {
    await page.setViewportSize({ width, height: 844 });
    const session = await request.post("/api/session");
    expect(session.ok()).toBeTruthy();
    const headers = { "x-local-lm-csrf": (await session.json() as { csrf_token: string }).csrf_token };
    const created = await request.post("/api/chats", { headers, data: { title: "Garden picture purposes" } });
    expect(created.status()).toBe(201);
    const { id } = await created.json() as { id: string };
    await page.addInitScript((chatId) => localStorage.setItem("local-lm-chat", chatId), id);
    await page.goto("/");
    const setup = page.getByRole("dialog", { name: "Set up LM Atelier" });
    await expect(setup).toBeVisible();
    await setup.getByRole("button", { name: "Not now" }).click();
    const chooser = page.waitForEvent("filechooser");
    await page.getByRole("button", { name: "Attach file" }).click();
    await (await chooser).setFiles([0, 1, 2].map((index) => ({
      name: `garden-${index}.png`, mimeType: "image/png", buffer: Buffer.concat([PICTURE, Buffer.from([index])]),
    })));
    const purposes = page.getByRole("combobox", { name: "Picture purpose", exact: true });
    await expect(purposes).toHaveCount(3);
    await purposes.nth(1).selectOption("edit_source");
    await expect(purposes.nth(0)).toHaveValue("reference");
    await expect(purposes.nth(1)).toHaveValue("edit_source");
    await expect(purposes.nth(2)).toHaveValue("reference");
    type Attachment = { artifact_id: string; image_role: string | null };
    async function storedAttachments(): Promise<Attachment[]> {
      const response = await request.get(`/api/chats/${id}/composer-draft`);
      expect(response.ok()).toBeTruthy();
      return (await response.json() as { attachments: Attachment[] }).attachments;
    }
    await expect.poll(async () => (await storedAttachments()).map((item) => item.image_role))
      .toEqual(["reference", "edit_source", "reference"]);
    const acceptedOrder = (await storedAttachments()).map((item) => item.artifact_id);
    await page.reload();
    await expect(purposes).toHaveCount(3);
    await expect(purposes.nth(1)).toHaveValue("edit_source");
    const bounds = await purposes.evaluateAll((elements) => elements.map((element) => {
      const control = element.getBoundingClientRect();
      const card = element.closest(".attachment-card")!.getBoundingClientRect();
      const strip = element.closest(".attachment-strip")!.getBoundingClientRect();
      return {
        controlFits: control.left >= card.left - 1 && control.right <= card.right + 1,
        cardFits: card.left >= strip.left - 1 && card.right <= strip.right + 1,
        stripFits: strip.left >= -1 && strip.right <= window.innerWidth + 1,
      };
    }));
    expect(bounds).toEqual(Array.from({ length: 3 }, () => ({ controlFits: true, cardFits: true, stripFits: true })));
    await purposes.nth(1).selectOption("automatic");
    await expect(purposes.nth(0)).toHaveValue("automatic");
    await expect(purposes.nth(1)).toHaveValue("automatic");
    await expect(purposes.nth(2)).toHaveValue("automatic");
    await expect.poll(async () => (await storedAttachments()).map((item) => item.image_role)).toEqual([null, null, null]);
    expect((await storedAttachments()).map((item) => item.artifact_id)).toEqual(acceptedOrder);
  });
}
