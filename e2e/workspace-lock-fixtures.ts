import { expect, type APIRequestContext, type Page } from "@playwright/test";

/** Leave the workspace unlocked with the lock off, since every spec shares one server. */
export async function openWorkspaceForOthers(request: APIRequestContext) {
  const session = await request.post("/api/session");
  expect(session.ok()).toBeTruthy();
  const { csrf_token: token } = await session.json() as { csrf_token: string };
  const headers = { "x-local-lm-csrf": token };
  const unlocked = await request.post("/api/privacy/unlock", { headers, data: {} });
  expect(unlocked.ok()).toBeTruthy();
  const policy = await request.get("/api/privacy/policy");
  expect(policy.ok()).toBeTruthy();
  const { enabled, revision } = await policy.json() as { enabled: boolean; revision: number };
  if (!enabled) return;
  const off = await request.put("/api/privacy/policy", {
    headers, data: { expected_revision: revision, enabled: false },
  });
  expect(off.ok()).toBeTruthy();
}

export function lockedHeading(page: Page) {
  return page.getByRole("heading", { name: "LM Atelier is locked" });
}
