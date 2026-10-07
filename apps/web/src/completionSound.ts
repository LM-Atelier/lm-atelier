import { useSyncExternalStore } from "react";

/** A short chime for finished work, for somebody who would rather hear it than read it.
 *
 * Off until asked for. It needs no browser permission, which is what makes it
 * separate from notifications rather than part of them: somebody who blocked
 * notifications can still choose a sound.
 *
 * The chime is built from two tones on the spot rather than shipped as a
 * recording, so there is nothing to load and nothing to go missing. A finished
 * run rises and a failed one falls, so the two can be told apart without looking.
 */
export const SOUND_WHEN_FINISHED_KEY = "local-lm-sound-when-finished";

export type FinishedSound = "completed" | "failed";

/** The two tones of each chime, in hertz, in the order they play. */
export const CHIME_TONES: Record<FinishedSound, readonly [number, number]> = {
  completed: [660, 880],
  failed: [440, 330],
};

export function soundWhenFinished(): boolean {
  try {
    return localStorage.getItem(SOUND_WHEN_FINISHED_KEY) === "on";
  } catch {
    return false;
  }
}

const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === SOUND_WHEN_FINISHED_KEY) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** Whether finished work plays a chime, re-rendering whoever reads it when that changes. */
export function useSoundWhenFinished(): boolean {
  return useSyncExternalStore(subscribe, soundWhenFinished, () => false);
}

type AudioContextConstructor = new () => AudioContext;

function audioContextConstructor(): AudioContextConstructor | undefined {
  const scope = globalThis as { AudioContext?: AudioContextConstructor; webkitAudioContext?: AudioContextConstructor };
  return scope.AudioContext ?? scope.webkitAudioContext;
}

// One context for the life of the page: browsers limit how many may exist, and
// one opened while somebody pressed a button is the one most likely to be
// allowed to play later.
let context: AudioContext | null = null;

/** Play the chime for a finished or failed run. Never throws.
 *
 * A browser may refuse to start audio on a page nobody has interacted with
 * since it loaded. Then nothing plays, which is the same as the setting being
 * off, rather than an error in the middle of a live update.
 */
export function playFinishedSound(kind: FinishedSound): void {
  try {
    const Context = audioContextConstructor();
    if (!Context) return;
    context ??= new Context();
    if (context.state === "suspended") void context.resume().catch(() => undefined);
    const startsAt = context.currentTime;
    CHIME_TONES[kind].forEach((frequency, index) => {
      const at = startsAt + index * 0.14;
      const tone = context!.createOscillator();
      const volume = context!.createGain();
      tone.type = "sine";
      tone.frequency.setValueAtTime(frequency, at);
      volume.gain.setValueAtTime(0.0001, at);
      volume.gain.exponentialRampToValueAtTime(0.18, at + 0.02);
      volume.gain.exponentialRampToValueAtTime(0.0001, at + 0.3);
      tone.connect(volume);
      volume.connect(context!.destination);
      tone.start(at);
      tone.stop(at + 0.32);
    });
  } catch {
    // Audio that cannot start is a missed chime, not a broken workspace.
  }
}

/** Turn the chime on or off. Turning it on plays it once, so the choice is heard.
 *
 * Returns false, and leaves it off, in a browser that cannot play audio at all,
 * so the setting never claims a sound it cannot make.
 */
export function setSoundWhenFinished(on: boolean): boolean {
  if (on && !audioContextConstructor()) {
    remember(false);
    return false;
  }
  remember(on);
  if (on) playFinishedSound("completed");
  return true;
}

function remember(on: boolean): void {
  try {
    localStorage.setItem(SOUND_WHEN_FINISHED_KEY, on ? "on" : "off");
  } catch {
    return;
  }
  for (const listener of listeners) listener();
}
