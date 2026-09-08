import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
  sessionStorage.clear();
  localStorage.clear();
});

it("sends the reference image review to its exact subject and attachment", async () => {
  const reviewed = { asset: { id: "asset/2", validation_state: "weak" }, width: 512, height: 512, review_version: 2 };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(reviewed), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  const body = { outcome: "weak" as const, reasons: ["Image is blurred"] };
  await expect(api.reviewReferenceAsset("subject/1", "asset/2", body)).resolves.toEqual(reviewed);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/references/subject%2F1/assets/asset%2F2/review");
  expect(fetchMock.mock.calls[1][1]?.method).toBe("POST");
  expect(JSON.parse(fetchMock.mock.calls[1][1]?.body as string)).toEqual(body);
});
