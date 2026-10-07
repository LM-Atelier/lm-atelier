import { expect, test } from "@playwright/test";

for (const width of [1280, 390]) {
  test(`organizes an exact selection across pages without copying media at ${width}px`, async ({ page, request }) => {
    test.setTimeout(150_000);
    await page.setViewportSize({ width, height: 900 });
    await page.route("**/api/setup/readiness", (route) => route.fulfill({ json: { version: 2, state: "ready", roles: [] } }));
    const { csrf_token } = await (await request.post("/api/session")).json() as { csrf_token: string };
    const headers = { "x-local-lm-csrf": csrf_token };
    const prefix = `Organization ${width}`;
    const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6V8AAAAASUVORK5CYII=", "base64");
    const artifacts: { id: string; bytes: Buffer }[] = [];
    for (let index = 1; index <= 21; index++) {
      const bytes = Buffer.concat([png, Buffer.from(`${prefix} ${index}`)]);
      const uploaded = await request.post("/api/artifacts?kind=image", { headers, multipart: {
        file: { name: `${prefix} ${index}.png`, mimeType: "image/png", buffer: bytes },
      } });
      expect(uploaded.status()).toBe(201);
      artifacts.push({ id: (await uploaded.json() as { id: string }).id, bytes });
    }
    await page.goto("/?view=media");
    await page.getByRole("textbox", { name: "Search media" }).fill(prefix);
    await page.getByRole("button", { name: "Albums and tags", exact: true }).click();
    let manager = page.getByRole("dialog", { name: "Albums and tags" });
    const albumName = `Studies ${width}`;
    // Lose only the response after the real creation has committed.
    let lostResponse = true;
    const creationKeys: string[] = [];
    await page.route("**/api/media-collections", async (route) => {
      if (route.request().method() !== "POST") return route.continue();
      creationKeys.push((route.request().postDataJSON() as { operation_key: string }).operation_key);
      const response = await route.fetch();
      expect(response.status()).toBe(201);
      if (lostResponse) { lostResponse = false; await route.abort("failed"); }
      else await route.fulfill({ response });
    });
    await manager.getByRole("textbox", { name: "New album name" }).fill(albumName);
    await manager.getByRole("button", { name: "Create album", exact: true }).click();
    await expect(manager.getByText("Creation could not be confirmed.", { exact: false })).toBeVisible();
    await manager.getByRole("button", { name: "Close albums and tags" }).click();
    await page.getByRole("button", { name: "Albums and tags", exact: true }).click();
    manager = page.getByRole("dialog", { name: "Albums and tags" });
    await expect(manager.getByRole("textbox", { name: "New album name" })).toHaveValue(albumName);
    await manager.getByRole("button", { name: "Try creating this album again" }).click();
    await expect(manager.getByRole("textbox", { name: "New album name" })).toHaveValue("");
    expect(creationKeys).toHaveLength(2);
    expect(creationKeys[1]).toBe(creationKeys[0]);
    const albums = await (await request.get("/api/media-collections")).json() as { collections: { id: string; name: string }[] };
    const matching = albums.collections.filter((album) => album.name === albumName);
    expect(matching).toHaveLength(1);
    const albumId = matching[0].id;
    const tagName = `Landscape ${width}`;
    const tagCreationKeys: string[] = [];
    await page.route("**/api/media-tags", async (route) => {
      if (route.request().method() === "POST") {
        tagCreationKeys.push((route.request().postDataJSON() as { operation_key: string }).operation_key);
      }
      await route.continue();
    });
    await manager.getByRole("textbox", { name: "New tag label" }).fill(`${tagName}!`);
    await manager.getByRole("button", { name: "Create tag", exact: true }).click();
    await expect(manager.getByText("Creation was refused. Edit the name and try again.")).toBeVisible();
    await expect(manager.getByRole("textbox", { name: "New tag label" })).toBeEditable();
    await manager.getByRole("textbox", { name: "New tag label" }).fill(tagName);
    await manager.getByRole("button", { name: "Create tag", exact: true }).click();
    await expect(manager.getByRole("textbox", { name: "New tag label" })).toHaveValue("");
    expect(tagCreationKeys).toHaveLength(2);
    expect(tagCreationKeys[1]).not.toBe(tagCreationKeys[0]);
    await manager.getByRole("button", { name: "Close albums and tags" }).click();
    await page.getByRole("checkbox", { name: `Select ${prefix} 21.png`, exact: true }).check();
    await page.getByRole("button", { name: "Load more", exact: true }).click();
    await page.getByRole("checkbox", { name: `Select ${prefix} 1.png`, exact: true }).check();
    await page.getByRole("combobox", { name: "Selection action" }).selectOption("add-to-album");
    await page.getByRole("combobox", { name: "Destination album" }).selectOption({ label: albumName });
    await page.getByRole("button", { name: "Review selection" }).click();
    const review = page.getByRole("dialog", { name: "Review library changes" });
    await expect(review.getByText("2 changes will be applied to this exact selection.")).toBeVisible();
    const albumPath = `/api/artifact-library?collection_id=${albumId}`;
    expect((await (await request.get(albumPath)).json() as { items: unknown[] }).items).toHaveLength(0);
    await review.getByRole("button", { name: "Apply changes", exact: true }).click();
    await expect(review).toHaveCount(0);
    await page.getByRole("combobox", { name: "Album", exact: true }).selectOption({ label: albumName });
    await expect(page.getByRole("checkbox", { name: /^Select Organization/ })).toHaveCount(2);
    await page.getByRole("checkbox", { name: `Select ${prefix} 21.png`, exact: true }).check();
    await page.getByRole("checkbox", { name: `Select ${prefix} 1.png`, exact: true }).check();
    await page.getByRole("combobox", { name: "Selection action" }).selectOption("reorder-album");
    await page.getByRole("combobox", { name: "Destination album" }).selectOption({ label: albumName });
    await page.getByRole("button", { name: `Move ${prefix} 1.png earlier`, exact: true }).click();
    await page.getByRole("button", { name: "Review selection" }).click();
    await review.getByRole("button", { name: "Apply changes", exact: true }).click();
    await expect(review).toHaveCount(0);
    await expect(page.locator(".media-grid .gallery-card strong")).toHaveText([`${prefix} 1.png`, `${prefix} 21.png`]);
    await page.getByRole("checkbox", { name: `Select ${prefix} 1.png`, exact: true }).check();
    await page.getByRole("combobox", { name: "Selection action" }).selectOption("add-tag");
    await page.getByRole("combobox", { name: "Destination tag" }).selectOption({ label: tagName });
    await page.getByRole("button", { name: "Review selection" }).click();
    await review.getByRole("button", { name: "Apply changes", exact: true }).click();
    await expect(review).toHaveCount(0);
    await page.getByRole("combobox", { name: "Tag", exact: true }).selectOption({ label: tagName });
    await expect(page.getByRole("checkbox", { name: /^Select Organization/ })).toHaveCount(1);
    await page.getByRole("combobox", { name: "Tag", exact: true }).selectOption("");
    await expect(page.locator(".media-grid .gallery-card strong")).toHaveText([`${prefix} 1.png`, `${prefix} 21.png`]);
    await page.reload();
    await page.getByRole("textbox", { name: "Search media" }).fill(prefix);
    await page.getByRole("combobox", { name: "Album", exact: true }).selectOption({ label: albumName });
    await expect(page.locator(".media-grid .gallery-card strong")).toHaveText([`${prefix} 1.png`, `${prefix} 21.png`]);
    await page.reload();
    await page.getByRole("textbox", { name: "Search media" }).fill(prefix);
    await page.getByRole("combobox", { name: "Album", exact: true }).selectOption({ label: albumName });
    await page.getByRole("combobox", { name: "Tag", exact: true }).selectOption({ label: tagName });
    await expect(page.getByRole("checkbox", { name: `Select ${prefix} 1.png`, exact: true })).toBeVisible();
    await expect(page.getByRole("checkbox", { name: /^Select Organization/ })).toHaveCount(1);
    for (const artifact of [artifacts[0], artifacts[20]]) {
      const response = await request.get(`/api/artifacts/${encodeURIComponent(artifact.id)}/content`);
      expect(response.status()).toBe(200);
      expect(await response.body()).toEqual(artifact.bytes);
    }
    await page.getByRole("button", { name: "Albums and tags", exact: true }).click();
    manager = page.getByRole("dialog", { name: "Albums and tags" });
    const destinationName = `Sketch ${width}`;
    await manager.getByRole("textbox", { name: "New tag label" }).fill(destinationName);
    await manager.getByRole("button", { name: "Create tag", exact: true }).click();
    await expect(manager.getByRole("textbox", { name: "New tag label" })).toHaveValue("");
    await manager.getByRole("combobox", { name: "Edit album or tag" }).selectOption({ label: tagName });
    await manager.getByRole("textbox", { name: "Tag label", exact: true }).fill(destinationName);
    await manager.getByRole("button", { name: "Review rename" }).click();
    await expect(manager.getByText("This tag name is already in use. Choose a different name.")).toBeVisible();
    await expect(manager.getByRole("textbox", { name: "Tag label", exact: true })).toBeEditable();
    await manager.getByRole("combobox", { name: "Merge into tag" }).selectOption({ label: destinationName });
    await manager.getByRole("button", { name: "Review tag merge" }).click();
    await review.getByRole("button", { name: "Apply changes", exact: true }).click();
    await expect(review).toHaveCount(0);
    await expect(page.getByRole("combobox", { name: "Tag", exact: true })).toHaveValue("");
    await expect(page.getByRole("combobox", { name: "Album", exact: true })).toHaveValue(albumId);
    await expect(page.locator(".media-grid .gallery-card strong")).toHaveText([`${prefix} 1.png`, `${prefix} 21.png`]);
    await page.getByRole("combobox", { name: "Tag", exact: true }).selectOption({ label: destinationName });
    await expect(page.getByRole("checkbox", { name: /^Select Organization/ })).toHaveCount(1);
    await page.getByRole("button", { name: "Albums and tags", exact: true }).click();
    manager = page.getByRole("dialog", { name: "Albums and tags" });
    await manager.getByRole("combobox", { name: "Edit album or tag" }).selectOption({ label: destinationName });
    await manager.getByRole("button", { name: "Review removal" }).click();
    await review.getByRole("button", { name: "Apply changes", exact: true }).click();
    await expect(review).toHaveCount(0);
    await expect(page.getByRole("combobox", { name: "Tag", exact: true })).toHaveValue("");
    await expect(page.locator(".media-grid .gallery-card strong")).toHaveText([`${prefix} 1.png`, `${prefix} 21.png`]);
    await page.getByRole("button", { name: "Albums and tags", exact: true }).click();
    manager = page.getByRole("dialog", { name: "Albums and tags" });
    await manager.getByRole("combobox", { name: "Edit album or tag" }).selectOption({ label: albumName });
    await manager.getByRole("button", { name: "Review removal" }).click();
    await review.getByRole("button", { name: "Apply changes", exact: true }).click();
    await expect(review).toHaveCount(0);
    await expect(page.getByRole("combobox", { name: "Album", exact: true })).toHaveValue("");
    await expect(page.getByRole("textbox", { name: "Search media" })).toHaveValue(prefix);
    await expect(page.getByRole("checkbox", { name: /^Select Organization/ })).toHaveCount(20);
    await expect(page.getByText("The Media Library could not be loaded safely. Refresh and try again.")).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)).toBe(false);
  });
}
