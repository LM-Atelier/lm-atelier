import { afterEach, expect, it, vi } from "vitest";
import { DEFAULT_OUTPUT_SHAPES, defaultedShape, setOutputShapeChoice } from "./outputShapePreferences";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
  sessionStorage.clear();
  localStorage.clear();
});

function answered(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

it("sends the browser's default shapes with a turn once one is chosen, and nothing before", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(answered(200, { csrf_token: "csrf" }))
    .mockResolvedValueOnce(answered(202, { accepted: true }))
    .mockResolvedValueOnce(answered(202, { accepted: true }));
  vi.stubGlobal("fetch", fetchMock);
  const { api } = await import("./api");

  await api.sendTurn("chat", "A plain gray square", "image", [], {}, "before-a-default");
  setOutputShapeChoice("image", defaultedShape(DEFAULT_OUTPUT_SHAPES.image, "3:2"));
  await api.sendTurn("chat", "A plain gray square", "image", [], {}, "with-a-default");

  const before = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
  const after = JSON.parse(String(fetchMock.mock.calls[2][1]?.body));
  expect(before).not.toHaveProperty("default_output_shapes");
  expect(after.default_output_shapes).toEqual({ image: "3:2", video: null });
  expect({ ...after, default_output_shapes: undefined, idempotency_key: "" })
    .toEqual({ ...before, default_output_shapes: undefined, idempotency_key: "" });
});
