import { useCallback, useEffect, useMemo, useState } from "react";

import {
  prunedAutoSettingsRoles,
  readAutoSettingsRoles,
  withAutoSettingsRole,
  writeAutoSettingsRoles,
} from "./autoSettingsRoles";
import type { EngineRole } from "./types";

/**
 * Owns which role tab each chat's settings drawer was left on.
 *
 * The state and the writing that keeps it are one concern, so they live
 * together rather than as a `useState` in the view with a `useEffect` beside
 * it. App.tsx is against its `max-lines` brake, and the brake is right here:
 * this is a self-contained rule about storage, not part of rendering a chat.
 */
export function useAutoSettingsRoles(
  knownChats: readonly { id: string }[] | undefined,
): [Record<string, EngineRole>, (chatId: string, role: EngineRole) => void] {
  const [roles, setRoles] = useState<Record<string, EngineRole>>(
    () => readAutoSettingsRoles(localStorage),
  );
  const knownChatIds = useMemo(
    () => (knownChats === undefined ? undefined : knownChats.map((chat) => chat.id)),
    [knownChats],
  );

  useEffect(() => {
    // Pruned against the live list so a deleted chat stops being remembered.
    // Readiness travels as `undefined`: before the query resolves nothing is
    // pruned, and once it resolves an empty list genuinely means no chats
    // exist and the record is cleared.
    writeAutoSettingsRoles(localStorage, prunedAutoSettingsRoles(roles, knownChatIds));
  }, [roles, knownChatIds]);

  // Stable identity, so callers can list it in a dependency array without
  // re-running their effects on every render.
  const remember = useCallback(
    (chatId: string, role: EngineRole) =>
      setRoles((current) => withAutoSettingsRole(current, chatId, role)),
    [],
  );
  return [roles, remember];
}
