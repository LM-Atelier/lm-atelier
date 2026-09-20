import { createContext, useContext, useSyncExternalStore } from "react";
import { CHAT_ACTIVITY_SEEN_KEY, createChatActivitySeenStore, type ChatActivitySeenStore } from "./chatActivitySeen";

export const ChatActivitySeenContext = createContext<ChatActivitySeenStore | null>(null);

function persistentStore(): ChatActivitySeenStore {
  let storage: Storage | null;
  try { storage = window.localStorage; } catch { storage = null; }
  const store = createChatActivitySeenStore(storage);
  let subscribers = 0;
  const refresh = (event: StorageEvent) => {
    if (event.storageArea === storage && (event.key === CHAT_ACTIVITY_SEEN_KEY || event.key === null)) {
      store.refresh(event.key === null ? null : event.newValue);
    }
  };
  return {
    ...store,
    subscribe: (listener) => {
      if (subscribers++ === 0) {
        try { if (storage) store.refresh(storage.getItem(CHAT_ACTIVITY_SEEN_KEY)); } catch { /* Keep in-memory state when storage becomes unavailable. */ }
        window.addEventListener("storage", refresh);
      }
      const unsubscribe = store.subscribe(listener);
      return () => {
        unsubscribe();
        if (--subscribers === 0) window.removeEventListener("storage", refresh);
      };
    },
  };
}

const defaultStore = persistentStore();

export function useChatActivitySeen() {
  const store = useContext(ChatActivitySeenContext) ?? defaultStore;
  useSyncExternalStore(store.subscribe, store.snapshot, store.snapshot);
  return store;
}
