import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { StrictMode, type PropsWithChildren } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAppNavigation } from "./useAppNavigation";
import { DEFAULT_SETTINGS_DESTINATION } from "./settingsDestinations";

beforeEach(() => {
  window.history.replaceState(null, "", "/");
  sessionStorage.clear();
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("normalizes unknown navigation while preserving unrelated URL values and history state", () => {
  const state = { launcher: "fixture" };
  window.history.replaceState(state, "", "/?view=unknown&settings=retired&keep=a&keep=b#anchor");
  const replace = vi.spyOn(window.history, "replaceState");
  const push = vi.spyOn(window.history, "pushState");
  const { result } = renderHook(useAppNavigation);
  expect(result.current.view).toBe("chat");
  expect(new URL(window.location.href).searchParams.getAll("keep")).toEqual(["a", "b"]);
  expect(new URL(window.location.href).searchParams.has("settings")).toBe(false);
  expect(window.location.hash).toBe("#anchor");
  expect(window.history.state).toEqual(state);
  expect(replace).toHaveBeenCalledTimes(1);
  expect(push).not.toHaveBeenCalled();
  expect(result.current.settingsFocusRequest).toBeUndefined();
  expect(result.current.mainFocusRequest).toBeUndefined();
});

it("opens a valid Settings deep link without requesting focus", () => {
  window.history.replaceState(null, "", "/?view=settings&settings=advanced");
  const { result } = renderHook(useAppNavigation);
  expect(result.current.view).toBe("settings");
  expect(result.current.settingsDestination).toBe("advanced");
  expect(result.current.settingsFocusRequest).toBeUndefined();
});

it("falls back for an invalid section rather than rendering an empty page", () => {
  window.history.replaceState(null, "", "/?view=settings&settings=retired");
  const { result } = renderHook(useAppNavigation);
  expect(result.current.settingsDestination).toBe(DEFAULT_SETTINGS_DESTINATION);
  expect(new URL(window.location.href).searchParams.get("settings")).toBe(DEFAULT_SETTINGS_DESTINATION);
});

it("retains the chosen section across view changes and a remount within the tab", () => {
  const mounted = renderHook(useAppNavigation);
  act(() => mounted.result.current.setSettingsDestination("advanced"));
  act(() => mounted.result.current.setView("workflows"));
  expect(new URL(window.location.href).searchParams.has("settings")).toBe(false);
  mounted.unmount();
  const next = renderHook(useAppNavigation);
  expect(next.result.current.view).toBe("workflows");
  act(() => next.result.current.setView("settings"));
  expect(next.result.current.settingsDestination).toBe("advanced");
  expect(next.result.current.settingsFocusRequest).toBeUndefined();
});

it("supports functional setters and consecutive navigation calls", () => {
  const { result } = renderHook(useAppNavigation);
  act(() => {
    result.current.setView("models");
    result.current.setView((previous) => previous === "models" ? "workflows" : "chat");
  });
  expect(result.current.view).toBe("workflows");
  expect(new URL(window.location.href).searchParams.get("view")).toBe("workflows");
});

it("refocuses a repeated section selection without adding a history entry", () => {
  const { result } = renderHook(useAppNavigation);
  act(() => result.current.setSettingsDestination("advanced"));
  const focus = result.current.settingsFocusRequest;
  const push = vi.spyOn(window.history, "pushState");
  act(() => result.current.setSettingsDestination("advanced"));
  expect(push).not.toHaveBeenCalled();
  expect(result.current.settingsFocusRequest).not.toBe(focus);
  act(() => result.current.setView("settings"));
  expect(push).not.toHaveBeenCalled();
  expect(result.current.settingsFocusRequest).toBeUndefined();
});

it("restores both page and section on actual Back and Forward", async () => {
  const { result } = renderHook(useAppNavigation);
  act(() => result.current.setSettingsDestination("advanced"));
  act(() => result.current.setView("workflows"));
  act(() => window.history.back());
  await waitFor(() => expect(result.current.view).toBe("settings"));
  expect(result.current.settingsDestination).toBe("advanced");
  expect(result.current.settingsFocusRequest).toBeDefined();
  expect(result.current.mainFocusRequest).toBeUndefined();
  act(() => window.history.back());
  await waitFor(() => expect(result.current.view).toBe("chat"));
  expect(result.current.mainFocusRequest).toBeDefined();
  act(() => window.history.forward());
  await waitFor(() => expect(result.current.view).toBe("settings"));
  expect(result.current.settingsDestination).toBe("advanced");
});

it("normalizes malformed history in place", () => {
  const { result } = renderHook(useAppNavigation);
  window.history.pushState({ fixture: true }, "", "/?view=invalid&settings=advanced&other=1");
  const push = vi.spyOn(window.history, "pushState");
  act(() => window.dispatchEvent(new PopStateEvent("popstate")));
  expect(result.current.view).toBe("chat");
  expect(new URL(window.location.href).searchParams.get("other")).toBe("1");
  expect(window.history.state).toEqual({ fixture: true });
  expect(push).not.toHaveBeenCalled();
});

it("does not duplicate normalization or listeners under StrictMode and remount", () => {
  function Wrapper({ children }: PropsWithChildren) { return <StrictMode>{children}</StrictMode>; }
  const replace = vi.spyOn(window.history, "replaceState");
  const first = renderHook(useAppNavigation, { wrapper: Wrapper });
  expect(replace).toHaveBeenCalledTimes(1);
  first.unmount();
  const next = renderHook(useAppNavigation, { wrapper: Wrapper });
  expect(replace).toHaveBeenCalledTimes(1);
  act(() => window.dispatchEvent(new PopStateEvent("popstate")));
  expect(next.result.current.mainFocusRequest).toBe(1);
});

it("continues navigating when session storage is unavailable", () => {
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("Unavailable"); });
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("Unavailable"); });
  const { result } = renderHook(useAppNavigation);
  act(() => result.current.setSettingsDestination("advanced"));
  act(() => result.current.setView("chat"));
  act(() => result.current.setView("settings"));
  expect(result.current.settingsDestination).toBe("advanced");
});
