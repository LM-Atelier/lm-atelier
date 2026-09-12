import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

/** The output-shape row, measured in a real browser at a real window size.
 *
 * This row's defect was a layout one: seven named choices were laid out past the
 * edge of the settings list, and six of them were unreachable at every window
 * width. That cannot be tested where the component tests live. jsdom loads no
 * stylesheet and lays nothing out - `document.styleSheets.length` is 0 and every
 * rectangle is zero - so `scrollWidth <= clientWidth` there reads `0 <= 0` and
 * passes whether the defect is present or not, which is a check comparing an
 * absent measurement with itself.
 *
 * Reaching this panel at all needed a workflow whose stored graph PROVES that a
 * declared width and height reach its output, because the shape control offers
 * nothing otherwise. That is what the graph below is: the smallest shape the
 * geometry proof accepts.
 */

const WORKFLOW_NAME = "Shape Certification Fixture";

/** This panel needs a TRUSTED revision, and trust needs a MANAGED engine.
 *
 * A review grants core trust only to node classes an engine reports, and the
 * engine is only asked when a media worker the product itself started is running
 * and ready. An engine started BESIDE the product is reachable but not managed,
 * so the ordinary browser run can never approve a synthetic workflow and the
 * shape control correctly offers nothing there. Only the managed-media runner
 * arranges it, so these stand down anywhere else rather than failing.
 */
function requiresTheManagedMediaRunner() {
  test.skip(
    !process.env.LM_ATELIER_E2E_BASE_URL || !process.env.LM_ATELIER_E2E_MANAGED_MEDIA,
    "requires the runner that lets the product start its own engine, which is what makes a revision trustable",
  );
}

async function dismissSetup(page: Page) {
  const setupDialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  await expect(setupDialog).toBeVisible();
  await setupDialog.getByRole("button", { name: "Not now" }).click();
  await expect(setupDialog).toBeHidden();
}

async function createSession(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/session");
  expect(response.ok()).toBeTruthy();
  const payload = await response.json() as { csrf_token: string };
  return payload.csrf_token;
}

/** A workflow the review will accept AND the geometry proof will bind to.
 *
 * The four-node spine the proof walks - SaveImage back through VAEDecode and
 * KSampler to exactly one latent root, whose width and height are the declared
 * placeholders - is not a workflow the REVIEW will pass. A review refuses any
 * node missing an input the engine declares required, so a KSampler with no
 * model and no conditioning is structurally invalid however the graph is walked
 * afterwards. Both have to hold at once, which is what this is.
 */
function apiGraph(): Record<string, unknown> {
  return {
    checkpoint: {
      class_type: "CheckpointLoaderSimple",
      inputs: { ckpt_name: "model.safetensors" },
    },
    positive: {
      class_type: "CLIPTextEncode",
      inputs: { text: "${prompt}", clip: ["checkpoint", 1] },
    },
    negative: {
      class_type: "CLIPTextEncode",
      inputs: { text: "${negative_prompt}", clip: ["checkpoint", 1] },
    },
    latent: {
      class_type: "EmptyLatentImage",
      inputs: { width: "${width}", height: "${height}", batch_size: 1 },
    },
    sampler: {
      class_type: "KSampler",
      inputs: {
        model: ["checkpoint", 0],
        seed: 1,
        steps: 4,
        cfg: 7.0,
        sampler_name: "euler",
        scheduler: "normal",
        positive: ["positive", 0],
        negative: ["negative", 0],
        latent_image: ["latent", 0],
        denoise: 1.0,
      },
    },
    decode: { class_type: "VAEDecode", inputs: { samples: ["sampler", 0], vae: ["checkpoint", 2] } },
    save: {
      class_type: "SaveImage",
      inputs: { images: ["decode", 0], filename_prefix: "certification" },
    },
  };
}

/** The schema shape the proof requires, in the shipped convention.
 *
 * Two things here are easy to get wrong and impossible to diagnose from the API,
 * because capability refusal is deliberately content-free - a workflow's graph
 * must not become an error oracle - so a wrong schema reads as a flat
 * "unsupported" with no reason attached.
 *
 * Width and height must carry ALL of type, default, minimum, maximum and
 * multipleOf: a keyword the proof does not read could narrow the range it is
 * about to advertise, so it refuses rather than advertise a size the workflow
 * would then reject.
 *
 * And `prompt` is a RESERVED key that a workflow may not declare as a setting.
 * Declaring it as an ordinary property makes the whole capability unavailable.
 * The shipped compiler declares every runtime parameter as `readOnly` instead,
 * which is what this does.
 */
function inputSchema(): Record<string, unknown> {
  return {
    type: "object",
    properties: {
      prompt: { readOnly: true },
      negative_prompt: { readOnly: true },
      width: { type: "integer", default: 1024, minimum: 128, maximum: 2048, multipleOf: 64 },
      height: { type: "integer", default: 768, minimum: 128, maximum: 2048, multipleOf: 64 },
    },
  };
}

/** Install the workflow AND get its revision trusted.
 *
 * Creation always stores an untrusted revision - `create_workflow` passes
 * `trusted=False` itself, so asking for trust in the payload is both ignored
 * and rejected, because request models forbid unknown fields. Trust comes from
 * the review endpoint, and the geometry proof refuses anything untrusted, so
 * without this second step the shape control correctly offers nothing and the
 * test would be measuring an empty panel.
 */
async function installShapeCapableWorkflow(
  request: APIRequestContext,
): Promise<{ familyId: string; csrfToken: string }> {
  const csrfToken = await createSession(request);
  const headers = { "x-local-lm-csrf": csrfToken };
  const created = await request.post("/api/workflows", {
    headers,
    data: {
      name: WORKFLOW_NAME,
      operation: "text_to_image",
      description: "Synthetic workflow used only for browser shape certification.",
      engine: "comfyui",
      engine_version: "0.28.0",
      ui_graph: { version: 0.4, nodes: [], links: [] },
      api_graph: apiGraph(),
      input_schema: inputSchema(),
      dependencies: {},
    },
  });
  expect(created.status(), await created.text()).toBe(201);
  const workflow = await created.json() as {
    id: string;
    family_id: string;
    current_revision_id: string;
  };

  const review = await request.get(
    `/api/workflows/${workflow.id}/revisions/${workflow.current_revision_id}/review`,
    { headers },
  );
  expect(review.status(), await review.text()).toBe(200);
  const snapshot = await review.json() as {
    subject_sha256: string;
    can_approve: boolean;
    reasons?: string[];
  };
  // Name the reasons in the failure. "can_approve was false" sends the next
  // person reading source; "workflow_review_node_unavailable" sends them to the
  // one line that decides it.
  expect(
    snapshot.can_approve,
    `the synthetic revision must be approvable; reasons: ${JSON.stringify(snapshot.reasons)}`,
  ).toBe(true);

  const approved = await request.post(
    `/api/workflows/${workflow.id}/revisions/${workflow.current_revision_id}/review`,
    { headers, data: { action: "approve", subject_sha256: snapshot.subject_sha256 } },
  );
  expect(approved.status(), await approved.text()).toBe(200);
  return { familyId: workflow.family_id, csrfToken };
}

/** A chat already set to make images with this workflow.
 *
 * Configured through the API rather than through the composer, and that is a
 * deliberate narrowing rather than a shortcut. What is being certified here is
 * the LAYOUT of the shape row, so the certification should depend on as little
 * else as possible - and the composer's two controls currently discard one
 * another's choice, which is its own defect and has its own row. Driving them
 * here would make this test fail for a reason that has nothing to do with the
 * thing it measures.
 */
async function chatReadyForShapes(
  request: APIRequestContext,
  csrfToken: string,
  familyId: string,
  title: string,
): Promise<string> {
  const headers = { "x-local-lm-csrf": csrfToken };
  const created = await request.post("/api/chats", { headers, data: { title } });
  expect(created.status(), await created.text()).toBe(201);
  const chat = await created.json() as { id: string };

  const mode = await request.patch(`/api/chats/${chat.id}`, {
    headers,
    data: { routing_mode: "image" },
  });
  expect(mode.status(), await mode.text()).toBe(200);

  const selection = await request.put(
    `/api/chats/${chat.id}/workflow-selections/image`,
    { headers, data: { mode: "family", workflow_family_id: familyId } },
  );
  expect(selection.status(), await selection.text()).toBe(200);
  return title;
}

/** Open that chat and its settings, through the interface a person uses.
 *
 * The sidebar is closed in a narrow window, so the way in is the navigation
 * toggle; it is clicked only when it is offered, which keeps one path for both
 * widths rather than two that could drift apart.
 */
async function openImageSettings(page: Page, chatTitle: string) {
  await page.goto("/");
  await dismissSetup(page);
  const navigation = page.getByRole("button", { name: "Toggle navigation" });
  if (await navigation.isVisible()) await navigation.click();
  await page.getByRole("button", { name: chatTitle, exact: true }).click();
  await page.getByRole("button", { name: "Turn settings" }).click();
  await expect(page.getByRole("group", { name: "Output aspect ratio" })).toBeVisible();
}

test("every output shape is reachable inside the settings list", async ({ page, request }) => {
  requiresTheManagedMediaRunner();
  const { familyId, csrfToken } = await installShapeCapableWorkflow(request);
  const title = await chatReadyForShapes(request, csrfToken, familyId, "Shapes at full width");
  await openImageSettings(page, title);

  const shapes = page.getByRole("group", { name: "Output aspect ratio" }).getByRole("button");
  const count = await shapes.count();
  // The instrument has to be shown to have moved: an empty group would satisfy
  // every assertion below while proving nothing at all.
  expect(count, "the workflow must offer more than one shape for this to mean anything").toBeGreaterThan(1);

  for (let index = 0; index < count; index += 1) {
    await expect(shapes.nth(index)).toBeInViewport();
  }

  const listFits = await page.locator(".settings-list").evaluate(
    (element) => element.scrollWidth <= element.clientWidth,
  );
  expect(listFits, "the shape row must not force the settings list to scroll sideways").toBe(true);
});

test("the shapes stay reachable in a narrow window", async ({ browser, request }) => {
  requiresTheManagedMediaRunner();
  const { familyId, csrfToken } = await installShapeCapableWorkflow(request);
  const title = await chatReadyForShapes(request, csrfToken, familyId, "Shapes at 320 pixels");
  // 320 CSS pixels is the width the reflow criterion names, and the width at
  // which the original defect hid six of the seven choices.
  const narrow = await browser.newContext({ viewport: { width: 320, height: 720 } });
  try {
    const page = await narrow.newPage();
    await openImageSettings(page, title);

    const shapes = page.getByRole("group", { name: "Output aspect ratio" }).getByRole("button");
    const count = await shapes.count();
    expect(count).toBeGreaterThan(1);
    for (let index = 0; index < count; index += 1) {
      await expect(shapes.nth(index)).toBeInViewport();
    }

    const listFits = await page.locator(".settings-list").evaluate(
      (element) => element.scrollWidth <= element.clientWidth,
    );
    expect(listFits, "a narrow window must not put the shapes outside the list").toBe(true);
  } finally {
    await narrow.close();
  }
});
