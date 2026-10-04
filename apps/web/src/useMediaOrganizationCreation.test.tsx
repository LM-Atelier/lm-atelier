import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { useMediaOrganizationCreation } from "./useMediaOrganizationCreation";

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return { ApiError: actual.ApiError, api: { createMediaCollection: vi.fn(), createMediaTag: vi.fn() } };
});
const album = { id: `collection_${"a".repeat(32)}`, kind: "manual", name: "Studies", description: "", version: 1 };

describe("catalog creation retries", () => {
  afterEach(cleanup);
  beforeEach(() => { vi.mocked(api.createMediaCollection).mockReset(); vi.mocked(api.createMediaTag).mockReset(); });
  it("reuses the original key after an uncertain response and rerender", async () => {
    const refresh = vi.fn();
    vi.mocked(api.createMediaCollection).mockRejectedValueOnce(new Error("connection-reset")).mockResolvedValueOnce(album);
    const hook = renderHook(() => useMediaOrganizationCreation(refresh));
    await act(async () => { expect(await hook.result.current.create("albums", "Studies")).toBe(false); });
    const key = hook.result.current.request?.operationKey;
    expect(key).toMatch(/^[0-9a-f]{32}$/);
    hook.rerender();
    await act(async () => { expect(await hook.result.current.create("albums", "Studies")).toBe(true); });
    expect(api.createMediaCollection).toHaveBeenNthCalledWith(1, "Studies", "", key);
    expect(api.createMediaCollection).toHaveBeenNthCalledWith(2, "Studies", "", key);
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(hook.result.current.request).toBeNull();
  });
  it("keeps a changed request from duplicating an uncertain write", async () => {
    vi.mocked(api.createMediaCollection).mockRejectedValue(new Error("connection-reset"));
    const hook = renderHook(() => useMediaOrganizationCreation(vi.fn()));
    await act(async () => { await hook.result.current.create("albums", "Studies"); });
    await act(async () => { expect(await hook.result.current.create("albums", "Different")).toBe(false); expect(await hook.result.current.create("tags", "Studies")).toBe(false); });
    expect(api.createMediaCollection).toHaveBeenCalledTimes(1);
    expect(api.createMediaTag).not.toHaveBeenCalled();
    expect(hook.result.current.request?.name).toBe("Studies");
  });
  it("retains the key when a successful response has the wrong original name", async () => {
    vi.mocked(api.createMediaCollection).mockResolvedValue({ ...album, name: "Different" });
    const refresh = vi.fn();
    const hook = renderHook(() => useMediaOrganizationCreation(refresh));
    await act(async () => { expect(await hook.result.current.create("albums", "Studies")).toBe(false); });
    expect(hook.result.current.request?.name).toBe("Studies");
    expect(refresh).not.toHaveBeenCalled();
  });

  it.each([
    ["albums", 422, "media-collection-invalid"],
    ["tags", 422, "media-tag-invalid"],
    ["tags", 409, "media-tag-conflict"],
  ] as const)("allows correction after a confirmed %s refusal %s %s", async (kind, status, code) => {
    const method = kind === "albums" ? vi.mocked(api.createMediaCollection) : vi.mocked(api.createMediaTag);
    method.mockRejectedValueOnce(new ApiError(status, null, "The request was refused.", code));
    const hook = renderHook(() => useMediaOrganizationCreation(vi.fn()));
    await act(async () => { expect(await hook.result.current.create(kind, "Studies")).toBe(false); });
    const originalKey = method.mock.calls[0][2];
    expect(hook.result.current.request).toBeNull();
    expect(hook.result.current.failed).toBe(true);
    method.mockResolvedValueOnce(kind === "albums" ? { ...album, name: "New studies" }
      : { id: `mediatag_${"b".repeat(32)}`, slug: "new-studies", label: "New studies", color: null, version: 1 });
    await act(async () => { expect(await hook.result.current.create(kind, "New studies")).toBe(true); });
    expect(method.mock.calls[1][2]).toMatch(/^[0-9a-f]{32}$/);
    expect(method.mock.calls[1][2]).not.toBe(originalKey);
  });
  it.each([
    [409, "media-creation-conflict"], [422, "media-collection-invalid"], [500, "media-tag-invalid"],
  ] as const)("retains a tag request on an ambiguous refusal %s %s", async (status, code) => {
    vi.mocked(api.createMediaTag).mockRejectedValueOnce(new ApiError(status, null, "The request was refused.", code));
    const hook = renderHook(() => useMediaOrganizationCreation(vi.fn()));
    await act(async () => { expect(await hook.result.current.create("tags", "Studies")).toBe(false); });
    expect(hook.result.current.request?.operationKey).toBe(vi.mocked(api.createMediaTag).mock.calls[0][2]);
    await act(async () => { expect(await hook.result.current.create("tags", "Different")).toBe(false); });
    expect(api.createMediaTag).toHaveBeenCalledTimes(1);
  });
});
