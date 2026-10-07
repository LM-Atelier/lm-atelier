import { expect, test, type Page } from "@playwright/test";

/** Correcting the perspective in a real browser: corners dragged on the canvas, and the picture kept.
 *
 * The corners are dragged with the mouse at whatever zoom the studio fits the
 * picture to, so each drag goes through the same screen-to-picture mapping a
 * person's does. What has to hold is that the kept picture is the size the
 * panel named, and that the card seen at an angle fills it.
 */

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

const WIDTH = 160;
const HEIGHT = 120;
/** The card's corners from the top left, clockwise, and the picture's own. */
const CARD: Array<[number, number]> = [[40, 16], [128, 29], [133, 107], [24, 93]];
const OWN: Array<[number, number]> = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]];

/** A card seen at an angle on a dark ground. */
async function cardAtAnAngle(page: Page): Promise<Buffer> {
  const dataUrl = await page.evaluate(({ card, width, height }) => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no canvas");
    context.fillStyle = "#202020";
    context.fillRect(0, 0, width, height);
    context.fillStyle = "#e6b428";
    context.beginPath();
    card.forEach(([x, y], index) => (index === 0 ? context.moveTo(x, y) : context.lineTo(x, y)));
    context.closePath();
    context.fill();
    return canvas.toDataURL("image/png");
  }, { card: CARD, width: WIDTH, height: HEIGHT });
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

/** Press at `from` and let go at `to`, both in the picture's own pixels. */
async function drag(page: Page, from: [number, number], to: [number, number]) {
  const box = await page.locator(".studio-canvas-layers canvas[data-layer='interaction']").boundingBox();
  if (!box) throw new Error("no canvas");
  const onScreen = ([x, y]: [number, number]) => [box.x + (x * box.width) / WIDTH, box.y + (y * box.height) / HEIGHT];
  const [startX, startY] = onScreen(from);
  const [endX, endY] = onScreen(to);
  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(endX, endY, { steps: 6 });
  await page.mouse.up();
}

test("squares up a card seen at an angle into the size the panel names", async ({ page }) => {
  await page.goto("/");
  await dismissSetup(page);
  await page.locator(".primary-nav").getByRole("button", { name: "Image Studio" }).click();
  await page.getByLabel("Choose an image to edit").setInputFiles({
    name: "card-at-an-angle.png",
    mimeType: "image/png",
    buffer: await cardAtAnAngle(page),
  });

  await page.getByRole("button", { name: "Correct the perspective" }).click();
  await expect(page.getByText("The corners start at the picture's own.")).toBeVisible();
  // A press takes the nearest corner and moves it as far as the pointer goes,
  // so each drag starts just inside a corner of the picture and travels to
  // the card's corner.
  for (const [index, [x, y]] of CARD.entries()) {
    const press: [number, number] = [OWN[index][0] === 0 ? 2 : WIDTH - 2, OWN[index][1] === 0 ? 2 : HEIGHT - 2];
    await drag(page, press, [press[0] + x - OWN[index][0], press[1] + y - OWN[index][1]]);
  }
  const named = page.getByText(/^Makes a picture \d+ by \d+ pixels\.$/);
  await expect(named).toBeVisible();
  const [, width, height] = (await named.textContent())!.match(/(\d+) by (\d+)/)!;

  await page.getByRole("button", { name: "Apply the correction" }).click();

  await expect.poll(() => page.evaluate(async () => {
    const { id } = JSON.parse(localStorage.getItem("local-lm-studio-session") ?? "{}") as { id?: string };
    const session = (await (await fetch(`/api/studio/sessions/${id}`)).json()) as {
      messages: { role: string; parts: { type: string; artifact_id: string | null; metadata_json?: { provenance?: { local_edit?: { operation?: string } } } }[] }[];
    };
    const answer = session.messages.filter((message) => message.role === "assistant").at(-1);
    if (answer?.parts.find((part) => part.type === "generation_metadata")?.metadata_json?.provenance?.local_edit?.operation !== "perspective") return null;
    const image = answer.parts.find((part) => part.type === "image");
    const bitmap = await createImageBitmap(await (await fetch(`/api/artifacts/${image?.artifact_id}/content`)).blob());
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    // Four pixels in from every side, where only the card can be.
    let ground = 0;
    for (let y = 4; y < bitmap.height - 4; y += 1) {
      for (let x = 4; x < bitmap.width - 4; x += 1) {
        const at = (y * bitmap.width + x) * 4;
        if (Math.abs(pixels[at] - 230) > 8 || Math.abs(pixels[at + 1] - 180) > 8 || Math.abs(pixels[at + 2] - 40) > 8) ground += 1;
      }
    }
    return `${bitmap.width} by ${bitmap.height}, ${ground} pixels of the ground inside`;
  })).toBe(`${width} by ${height}, 0 pixels of the ground inside`);
});
