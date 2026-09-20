import { useEffect, type RefObject } from "react";
import type { ChatActivityReference } from "./types";
import { useChatActivitySeen } from "./useChatActivitySeen";

function reference(element: HTMLElement): ChatActivityReference | null {
  try {
    const value: unknown = JSON.parse(element.dataset.chatActivity ?? "null");
    if (!value || typeof value !== "object" || !("id" in value) || !("sequence" in value)
      || !("message_id" in value) || !("response_revision_id" in value) || !("occurred_at" in value)
      || typeof value.id !== "string" || typeof value.sequence !== "number"
      || typeof value.message_id !== "string" || typeof value.response_revision_id !== "string"
      || typeof value.occurred_at !== "string") return null;
    return value as ChatActivityReference;
  } catch { return null; }
}

function visible(element: HTMLElement, viewport: HTMLElement): boolean {
  if (document.visibilityState !== "visible" || !element.isConnected
    || document.querySelector('[role="dialog"][aria-modal="true"]')) return false;
  const surface = element.querySelector("img, video, audio") ?? element;
  for (let ancestor: Element | null = surface; ancestor; ancestor = ancestor.parentElement) {
    if (getComputedStyle(ancestor).opacity === "0") return false;
  }
  const bounds = surface.getBoundingClientRect();
  const root = viewport.getBoundingClientRect();
  const left = Math.max(bounds.left, root.left, 0);
  const right = Math.min(bounds.right, root.right, window.innerWidth);
  const top = Math.max(bounds.top, root.top, 0);
  const bottom = Math.min(bounds.bottom, root.bottom, window.innerHeight);
  if (right - left < 1 || bottom - top < Math.min(8, bounds.height) || bottom <= top) return false;
  for (const image of element.querySelectorAll("img")) {
    if (!image.complete || image.naturalWidth === 0) return false;
  }
  for (const media of element.querySelectorAll("video, audio")) {
    if (media instanceof HTMLMediaElement && media.readyState < 2) return false;
  }
  const hit = document.elementFromPoint((left + right) / 2, (top + bottom) / 2);
  return hit !== null && element.contains(hit);
}

/** Acknowledge only identities attached to content inside the live transcript. */
export function useVisibleChatActivity(viewport: RefObject<HTMLDivElement | null>, chatId: string | undefined) {
  const store = useChatActivitySeen();
  useEffect(() => {
    const root = viewport.current;
    if (!root || !chatId || typeof IntersectionObserver === "undefined") return;
    const observed = new Set<HTMLElement>();
    let active = true;
    let frame: number | null = null;
    const check = (element: HTMLElement) => {
      if (!active) return;
      const activity = reference(element);
      if (activity && !store.hasSeen(chatId, activity) && root.contains(element) && visible(element, root)) {
        store.markSeen(chatId, activity);
      }
    };
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) if (entry.isIntersecting && entry.target instanceof HTMLElement) check(entry.target);
    }, { root });
    const reconcile = () => {
      frame = null;
      for (const element of observed) {
        if (!root.contains(element)) { observer.unobserve(element); observed.delete(element); }
      }
      for (const element of root.querySelectorAll<HTMLElement>("[data-chat-activity]")) {
        if (!observed.has(element)) { observed.add(element); observer.observe(element); }
        check(element);
      }
    };
    const schedule = () => { if (frame === null) frame = requestAnimationFrame(reconcile); };
    const mutations = new MutationObserver(schedule);
    mutations.observe(document.documentElement, { childList: true, subtree: true, attributes: true,
      attributeFilter: ["data-chat-activity", "aria-modal", "hidden", "open", "style", "class"] });
    root.addEventListener("scroll", schedule, { passive: true });
    root.addEventListener("load", schedule, true);
    root.addEventListener("loadeddata", schedule, true);
    document.addEventListener("visibilitychange", schedule);
    window.addEventListener("resize", schedule);
    schedule();
    return () => {
      active = false;
      if (frame !== null) cancelAnimationFrame(frame);
      observer.disconnect();
      mutations.disconnect();
      root.removeEventListener("scroll", schedule);
      root.removeEventListener("load", schedule, true);
      root.removeEventListener("loadeddata", schedule, true);
      document.removeEventListener("visibilitychange", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [chatId, store, viewport]);
}
