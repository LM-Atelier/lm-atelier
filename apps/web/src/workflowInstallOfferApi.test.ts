import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
  sessionStorage.clear();
  localStorage.clear();
});

it("queues only the encoded server offer identity without browser plan or graph claims", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response("[]", { status: 202 }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");
  await expect(api.installWorkflowOffer("offer/one?candidate")).resolves.toEqual([]);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/workflow-install-offers/offer%2Fone%3Fcandidate/install");
  expect(fetchMock.mock.calls[1][1]?.method).toBe("POST");
  expect(fetchMock.mock.calls[1][1]?.body).toBeUndefined();
});
