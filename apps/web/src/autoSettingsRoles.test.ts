import { describe, expect, it } from "vitest";

import {
  AUTO_SETTINGS_ROLES_KEY,
  AUTO_SETTINGS_ROLES_VERSION,
  MAX_REMEMBERED_ROLES,
  prunedAutoSettingsRoles,
  readAutoSettingsRoles,
  withAutoSettingsRole,
  writeAutoSettingsRoles,
} from "./autoSettingsRoles";

function storageOf(value: string | null) {
  return { getItem: () => value };
}

describe("readAutoSettingsRoles", () => {
  it("keeps the pairs it recognises and drops only the junk", () => {
    const stored = JSON.stringify({
      version: AUTO_SETTINGS_ROLES_VERSION,
      roles: {
        good: "image",
        alsoGood: "video",
        notARole: "banana",
        numeric: 3,
        nested: { role: "chat" },
        "": "chat",
      },
    });
    expect(readAutoSettingsRoles(storageOf(stored))).toEqual({
      good: "image",
      alsoGood: "video",
    });
  });

  it("returns nothing rather than throwing for anything unparseable", () => {
    // Storage is shared with every other script on the origin and outlives
    // upgrades, so its contents are input. None of these should reach the UI
    // as an exception.
    for (const raw of [null, "", "not json", "[]", '"a string"', "17", "null"]) {
      expect(readAutoSettingsRoles(storageOf(raw))).toEqual({});
    }
  });

  it("refuses an unknown version even when its entries look like roles", () => {
    // The case a bare object could not defend against: a future build writes a
    // different meaning under the same key, and its chat-id-to-role-looking
    // pairs would otherwise be adopted as history.
    const future = JSON.stringify({
      version: AUTO_SETTINGS_ROLES_VERSION + 1,
      roles: { a: "image", b: "video" },
    });
    expect(readAutoSettingsRoles(storageOf(future))).toEqual({});

    const versionless = JSON.stringify({ a: "image" });
    expect(readAutoSettingsRoles(storageOf(versionless))).toEqual({});
  });

  it("survives a browser that refuses to hand over site data", () => {
    const hostile = {
      getItem: () => {
        throw new DOMException("denied", "SecurityError");
      },
    };
    expect(readAutoSettingsRoles(hostile)).toEqual({});
  });
});

describe("writeAutoSettingsRoles", () => {
  it("writes a versioned envelope under the shared key", () => {
    const written: Record<string, string> = {};
    writeAutoSettingsRoles({ setItem: (key, value) => { written[key] = value; } }, { a: "image" });
    expect(JSON.parse(written[AUTO_SETTINGS_ROLES_KEY])).toEqual({
      version: AUTO_SETTINGS_ROLES_VERSION,
      roles: { a: "image" },
    });
  });

  it("swallows a quota failure, because forgetting a tab is not an error", () => {
    const full = {
      setItem: () => {
        throw new DOMException("quota", "QuotaExceededError");
      },
    };
    expect(() => writeAutoSettingsRoles(full, { a: "chat" })).not.toThrow();
  });
});

describe("prunedAutoSettingsRoles", () => {
  it("drops chats that no longer exist", () => {
    const roles = { kept: "image", gone: "video" } as const;
    expect(prunedAutoSettingsRoles(roles, ["kept", "other"])).toEqual({ kept: "image" });
  });

  it("prunes nothing before the list has loaded, and clears once it loads empty", () => {
    // These are two different states and an earlier version collapsed them
    // into one empty array, which protected the cold start at the cost of
    // never pruning for somebody who really had deleted every chat.
    const roles = { a: "image", b: "video" } as const;
    expect(prunedAutoSettingsRoles(roles, undefined)).toEqual(roles);
    expect(prunedAutoSettingsRoles(roles, [])).toEqual({});
  });

  it("bounds the record even when every chat is still live", () => {
    const roles: Record<string, "chat"> = {};
    for (let index = 0; index < MAX_REMEMBERED_ROLES + 40; index += 1) {
      roles[`chat-${index}`] = "chat";
    }
    const pruned = prunedAutoSettingsRoles(roles, Object.keys(roles));
    expect(Object.keys(pruned)).toHaveLength(MAX_REMEMBERED_ROLES);
    // The newest survive: the ceiling must not evict what is in active use.
    expect(pruned[`chat-${MAX_REMEMBERED_ROLES + 39}`]).toBe("chat");
    expect(pruned["chat-0"]).toBeUndefined();
  });
});

describe("withAutoSettingsRole", () => {
  it("moves a re-picked chat to the end so the ceiling evicts the stalest", () => {
    // `{...roles, [id]: role}` would keep the original position, and the
    // ceiling would then drop the chat the reader uses most.
    const roles = { first: "chat", second: "image" } as const;
    const next = withAutoSettingsRole(roles, "first", "video");
    expect(Object.keys(next)).toEqual(["second", "first"]);
    expect(next.first).toBe("video");
  });

  it("does not mutate the record it was given", () => {
    const roles = { a: "chat" } as const;
    withAutoSettingsRole(roles, "b", "image");
    expect(roles).toEqual({ a: "chat" });
  });
});
