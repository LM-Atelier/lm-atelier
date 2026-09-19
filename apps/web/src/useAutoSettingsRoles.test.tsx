import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";

import { readAutoSettingsRoles, writeAutoSettingsRoles } from "./autoSettingsRoles";
import { useAutoSettingsRoles } from "./useAutoSettingsRoles";

beforeEach(() => {
  localStorage.clear();
  writeAutoSettingsRoles(localStorage, { recent: "chat", older: "video" });
});
afterEach(cleanup);

it("keeps off-page role memory until the full chat list is known", () => {
  const { rerender } = renderHook(
    ({ chats, complete }) => useAutoSettingsRoles(chats, { complete }),
    { initialProps: { chats: [{ id: "recent" }], complete: false } },
  );
  expect(readAutoSettingsRoles(localStorage)).toEqual({ recent: "chat", older: "video" });

  rerender({ chats: [], complete: false });
  expect(readAutoSettingsRoles(localStorage)).toEqual({ recent: "chat", older: "video" });

  rerender({ chats: [{ id: "recent" }], complete: true });
  expect(readAutoSettingsRoles(localStorage)).toEqual({ recent: "chat" });
});

it("persists a changed role for a chat outside the loaded page", () => {
  const { result } = renderHook(() => useAutoSettingsRoles([{ id: "recent" }], { complete: false }));
  act(() => result.current[1]("older", "image"));
  expect(readAutoSettingsRoles(localStorage)).toEqual({ recent: "chat", older: "image" });
});

it("still clears deleted chat memory when a complete list is empty", () => {
  renderHook(() => useAutoSettingsRoles([]));
  expect(readAutoSettingsRoles(localStorage)).toEqual({});
});
