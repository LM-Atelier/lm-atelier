import type { EngineRole } from "./types";

/**
 * Per-chat memory of which role tab the settings drawer was left on.
 *
 * Kept as pure functions over a plain record so the rules can be tested
 * without a browser: what survives a reload, what is discarded as junk, and
 * what is dropped when a chat goes away.
 */

export const AUTO_SETTINGS_ROLES_KEY = "lm-atelier-auto-settings-roles";

/**
 * The stored shape carries its own version.
 *
 * Without one, a future build writing a different shape under the same key
 * would have any role-looking entries in it accepted, because a bare object
 * of chat ids to roles is indistinguishable from a bare object of chat ids to
 * roles that meant something else. An unrecognised version is treated as no
 * history rather than as history to salvage.
 */
export const AUTO_SETTINGS_ROLES_VERSION = 1;

/**
 * A ceiling so the entry can never grow without bound.
 *
 * Pruning against the live chat list is the real mechanism, but it only runs
 * while chats are loaded. A reader that opens the app offline, or before the
 * list arrives, would otherwise carry every chat it has ever opened forever.
 */
export const MAX_REMEMBERED_ROLES = 200;

const ROLES: readonly EngineRole[] = ["chat", "image", "video"];

function isRole(value: unknown): value is EngineRole {
  return typeof value === "string" && (ROLES as readonly string[]).includes(value);
}

/**
 * Read the remembered roles, discarding anything that is not one.
 *
 * Storage is shared with every other script on the origin and survives
 * upgrades, so its contents are input rather than state: a value written by an
 * older build, a half-written entry, or a hand-edited one all arrive here. A
 * single bad pair drops that pair, not the whole record - losing one chat's
 * tab is a smaller harm than resetting everybody's.
 */
export function readAutoSettingsRoles(storage: Pick<Storage, "getItem">): Record<string, EngineRole> {
  let raw: string | null;
  try {
    raw = storage.getItem(AUTO_SETTINGS_ROLES_KEY);
  } catch {
    return {};
  }
  if (!raw) return {};
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return {};
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
  const envelope = parsed as { version?: unknown; roles?: unknown };
  if (envelope.version !== AUTO_SETTINGS_ROLES_VERSION) return {};
  const stored = envelope.roles;
  if (typeof stored !== "object" || stored === null || Array.isArray(stored)) return {};
  const roles: Record<string, EngineRole> = {};
  for (const [chatId, role] of Object.entries(stored as Record<string, unknown>)) {
    if (chatId && isRole(role)) roles[chatId] = role;
  }
  return roles;
}

/**
 * Persist the remembered roles, and never let a storage failure reach the UI.
 *
 * Writing can throw for reasons that have nothing to do with this feature -
 * a private window, a full quota, a browser configured to refuse site data.
 * Forgetting which tab a drawer was on is not worth an error boundary.
 */
export function writeAutoSettingsRoles(
  storage: Pick<Storage, "setItem">,
  roles: Record<string, EngineRole>,
): void {
  try {
    storage.setItem(
      AUTO_SETTINGS_ROLES_KEY,
      JSON.stringify({ version: AUTO_SETTINGS_ROLES_VERSION, roles }),
    );
  } catch {
    /* remembering a tab is a convenience, not a guarantee */
  }
}

/**
 * Drop chats that no longer exist, then bound what is left.
 *
 * Readiness is carried by `undefined`, not by emptiness. An earlier version
 * treated every empty array as "not loaded", which protected the cold start but
 * meant somebody who had genuinely deleted all their chats never had stale
 * entries removed - the two states are different and collapsing them made one
 * of them unreachable. `undefined` prunes nothing; an empty ARRAY is a loaded
 * list with no chats in it and clears the record.
 *
 * The ceiling keeps the newest entries, which are the ones at the end of the
 * record: `set` writes a chat by deleting and re-inserting it, so insertion
 * order is recency order.
 */
export function prunedAutoSettingsRoles(
  roles: Record<string, EngineRole>,
  knownChatIds: readonly string[] | undefined,
): Record<string, EngineRole> {
  let entries = Object.entries(roles);
  if (knownChatIds !== undefined) {
    const known = new Set(knownChatIds);
    entries = entries.filter(([chatId]) => known.has(chatId));
  }
  if (entries.length > MAX_REMEMBERED_ROLES) {
    entries = entries.slice(entries.length - MAX_REMEMBERED_ROLES);
  }
  return Object.fromEntries(entries);
}

/**
 * Record one chat's role, keeping insertion order meaningful.
 *
 * The existing key is removed before the new one is written so a re-picked
 * chat moves to the end. Without that, `{...roles, [chatId]: role}` would keep
 * the original position and the ceiling above would evict the chat the reader
 * uses most.
 */
export function withAutoSettingsRole(
  roles: Record<string, EngineRole>,
  chatId: string,
  role: EngineRole,
): Record<string, EngineRole> {
  const next = { ...roles };
  delete next[chatId];
  next[chatId] = role;
  return next;
}
