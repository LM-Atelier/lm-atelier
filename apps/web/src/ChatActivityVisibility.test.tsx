import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useRef, type ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ChatActivityIndicators } from "./ChatActivityIndicators";
import { createChatActivitySeenStore } from "./chatActivitySeen";
import { ChatActivitySeenContext } from "./useChatActivitySeen";
import { useVisibleChatActivity } from "./useVisibleChatActivity";
import { VisibleChatActivity } from "./VisibleChatActivity";
import type { ChatActivityReference } from "./types";

const first: ChatActivityReference = { id: "first", sequence: 1, message_id: "message", response_revision_id: "revision", occurred_at: "2026-09-20T00:00:00Z" };
const later: ChatActivityReference = { ...first, id: "later", sequence: 2 };
let callbacks: IntersectionObserverCallback[];

beforeEach(() => {
  callbacks = [];
  vi.stubGlobal("IntersectionObserver", class {
    constructor(callback: IntersectionObserverCallback) { callbacks.push(callback); }
    observe() {}
    unobserve() {}
    disconnect() {}
  });
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    return new DOMRect(0, this.dataset.offscreen ? 2_000 : 0, 400, 100);
  });
  Object.defineProperty(document, "elementFromPoint", { configurable: true, value: () => document.querySelector("[data-chat-activity]") });
  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  Reflect.deleteProperty(document, "elementFromPoint");
  Reflect.deleteProperty(document, "visibilityState");
});

function Transcript({ activity = first, children = "Authoritative output", chatId = "chat", offscreen = false }: {
  activity?: ChatActivityReference | null; children?: ReactNode; chatId?: string; offscreen?: boolean;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  useVisibleChatActivity(viewport, chatId);
  return <div ref={viewport} data-testid="transcript"><div data-offscreen={offscreen ? "true" : undefined}><VisibleChatActivity activity={activity}>{children}</VisibleChatActivity></div></div>;
}

function entry(target: HTMLElement): IntersectionObserverEntry {
  const bounds = target.getBoundingClientRect();
  return { target, isIntersecting: true, boundingClientRect: bounds, intersectionRect: bounds,
    rootBounds: bounds, intersectionRatio: 1, time: 0 };
}

function intersect(element = document.querySelector<HTMLElement>("[data-chat-activity]")!) {
  act(() => { for (const callback of callbacks) callback([entry(element)], {} as IntersectionObserver); });
}

it("keeps newer output unread when an older authoritative snapshot becomes visible", () => {
  const store = createChatActivitySeenStore(null);
  render(<ChatActivitySeenContext.Provider value={store}><Transcript /><ChatActivityIndicators chatId="chat" activity={{ active_work_count: 2, unresolved_failed_count: 1, last_output: later, last_failure: null }} /></ChatActivitySeenContext.Provider>);
  intersect();
  expect(store.hasSeen("chat", first)).toBe(true);
  expect(store.hasSeen("chat", later)).toBe(false);
  expect(screen.getByRole("img", { name: "Unread output" })).toBeVisible();
  expect(screen.getByRole("img", { name: "2 active tasks" })).toBeVisible();
  expect(screen.getByRole("img", { name: "1 unresolved failure" })).toBeVisible();
});

it("does not acknowledge a hidden document and reconciles when it becomes visible", async () => {
  const store = createChatActivitySeenStore(null);
  Object.defineProperty(document, "visibilityState", { configurable: true, value: "hidden" });
  render(<ChatActivitySeenContext.Provider value={store}><Transcript /></ChatActivitySeenContext.Provider>);
  intersect();
  expect(store.hasSeen("chat", first)).toBe(false);
  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
  fireEvent(document, new Event("visibilitychange"));
  await waitFor(() => expect(store.hasSeen("chat", first)).toBe(true));
});

it("rejects offscreen content and content outside the transcript", () => {
  const store = createChatActivitySeenStore(null);
  render(<ChatActivitySeenContext.Provider value={store}><Transcript activity={null} /><VisibleChatActivity activity={first}>Branch preview</VisibleChatActivity></ChatActivitySeenContext.Provider>);
  intersect();
  expect(store.hasSeen("chat", first)).toBe(false);
});

it("requires content to intersect the viewport even if an observer callback is stale", () => {
  const store = createChatActivitySeenStore(null);
  render(<ChatActivitySeenContext.Provider value={store}><Transcript /></ChatActivitySeenContext.Provider>);
  const element = document.querySelector<HTMLElement>("[data-chat-activity]")!;
  element.dataset.offscreen = "true";
  intersect(element);
  expect(store.hasSeen("chat", first)).toBe(false);
});

it("waits for a covering dialog to close", async () => {
  const store = createChatActivitySeenStore(null);
  const view = (dialog: boolean) => <ChatActivitySeenContext.Provider value={store}><Transcript />{dialog && <div role="dialog" aria-modal="true">Settings</div>}</ChatActivitySeenContext.Provider>;
  const { rerender } = render(view(true));
  intersect();
  expect(store.hasSeen("chat", first)).toBe(false);
  rerender(view(false));
  await waitFor(() => expect(store.hasSeen("chat", first)).toBe(true));
});

it("leaves a broken image unread and acknowledges only after actual image loading", async () => {
  const store = createChatActivitySeenStore(null);
  render(<ChatActivitySeenContext.Provider value={store}><Transcript><img alt="Color study" src="/neutral.png" /></Transcript></ChatActivitySeenContext.Provider>);
  intersect();
  expect(store.hasSeen("chat", first)).toBe(false);
  const picture = screen.getByRole("img", { name: "Color study" });
  Object.defineProperty(picture, "complete", { value: true, configurable: true });
  Object.defineProperty(picture, "naturalWidth", { value: 100, configurable: true });
  fireEvent.load(picture);
  await waitFor(() => expect(store.hasSeen("chat", first)).toBe(true));
});

it("ignores an old observer callback after switching chats", () => {
  const store = createChatActivitySeenStore(null);
  const { rerender } = render(<ChatActivitySeenContext.Provider value={store}><Transcript activity={null} /></ChatActivitySeenContext.Provider>);
  const old = callbacks[0];
  rerender(<ChatActivitySeenContext.Provider value={store}><Transcript activity={later} chatId="other-chat" /></ChatActivitySeenContext.Provider>);
  const element = document.querySelector<HTMLElement>("[data-chat-activity]")!;
  act(() => old([entry(element)], {} as IntersectionObserver));
  expect(store.hasSeen("chat", later)).toBe(false);
});

it.each(["output", "ancestor"])("leaves transparent %s unread until it becomes visible", async (target) => {
  const store = createChatActivitySeenStore(null);
  render(<ChatActivitySeenContext.Provider value={store}><Transcript /></ChatActivitySeenContext.Provider>);
  const output = document.querySelector<HTMLElement>("[data-chat-activity]")!;
  const element = target === "output" ? output : screen.getByTestId("transcript");
  element.style.opacity = "0";
  intersect(output);
  expect(store.hasSeen("chat", first)).toBe(false);
  element.style.opacity = "1";
  await waitFor(() => expect(store.hasSeen("chat", first)).toBe(true));
});
