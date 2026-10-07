import { useSyncExternalStore } from "react";

/** Which keystroke sends what has been typed.
 *
 * "enter" is how every text field that sends has always behaved: Enter sends
 * and Shift+Enter starts a new line. "mod-enter" is for somebody who writes
 * longer messages and would rather Enter only ever start a new line: then
 * Ctrl+Enter sends, or Cmd+Enter on a Mac.
 */
export type SendKeyChoice = "enter" | "mod-enter";

export const SEND_KEY_CHOICES: readonly SendKeyChoice[] = ["enter", "mod-enter"];

export const SEND_KEY_KEY = "local-lm-send-key";

export function isSendKeyChoice(value: unknown): value is SendKeyChoice {
  return typeof value === "string" && (SEND_KEY_CHOICES as readonly string[]).includes(value);
}

/** The choice as it stands at this moment.
 *
 * A keystroke is judged when it happens, so a text field reads this then rather
 * than subscribing: it cannot be out of date, and the field does not re-render
 * when somebody changes the setting.
 */
export function currentSendKeyChoice(): SendKeyChoice {
  return storedSendKeyChoice();
}

function storedSendKeyChoice(): SendKeyChoice {
  try {
    const stored = localStorage.getItem(SEND_KEY_KEY);
    return isSendKeyChoice(stored) ? stored : "enter";
  } catch {
    return "enter";
  }
}

const listeners = new Set<() => void>();

/** Remember which keystroke sends, and tell every text field that sends. */
export function setSendKeyChoice(choice: SendKeyChoice): void {
  try {
    localStorage.setItem(SEND_KEY_KEY, choice);
  } catch {
    // Without storage the choice cannot be remembered, and every field would go
    // on reading the old one, so nothing is told.
    return;
  }
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === SEND_KEY_KEY) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** The current send-key choice, re-rendering whoever reads it when it changes. */
export function useSendKeyChoice(): SendKeyChoice {
  return useSyncExternalStore(subscribe, storedSendKeyChoice, () => "enter");
}

type Keystroke = Pick<KeyboardEvent, "key" | "shiftKey" | "ctrlKey" | "metaKey"> & {
  keyCode?: number;
  nativeEvent?: { isComposing?: boolean };
};

/** Whether this keystroke sends, under the given choice.
 *
 * Shift+Enter never sends under either choice, because it is the new line
 * people already reach for everywhere else.
 *
 * Nor does Enter while text is still being composed. An input method - for
 * Japanese, Chinese or Korean, among others - uses Enter to confirm the word it
 * is building, and treating that as "send" would send half a sentence. Browsers
 * mark that keystroke as composing, and some older ones report it only as key
 * code 229.
 */
export function isSendKeystroke(event: Keystroke, choice: SendKeyChoice): boolean {
  if (event.key !== "Enter" || event.shiftKey) return false;
  if (event.nativeEvent?.isComposing || event.keyCode === 229) return false;
  return choice === "enter" || event.ctrlKey || event.metaKey;
}
