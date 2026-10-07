import { vi } from "vitest";

/** What a stand-in browser's audio did: the tones it started, and whether it was asked to resume. */
export interface HeardAudio {
  tones: number[];
  resumed: number;
  state: AudioContextState;
  /** When true, building a tone throws, as a browser that refuses audio might. */
  refuse: boolean;
}

// Shared by every stand-in context, because the page keeps the first context
// it opens for its whole life and a later test must still hear through it.
const heard: HeardAudio = { tones: [], resumed: 0, state: "running", refuse: false };

class StandInAudioContext {
  currentTime = 0;
  destination = {};

  get state(): AudioContextState {
    return heard.state;
  }

  resume(): Promise<void> {
    heard.resumed += 1;
    heard.state = "running";
    return Promise.resolve();
  }

  createOscillator() {
    if (heard.refuse) throw new DOMException("audio is not allowed", "NotAllowedError");
    const frequency = { value: 0, setValueAtTime: (value: number) => { frequency.value = value; } };
    return {
      type: "sine",
      frequency,
      connect: () => undefined,
      start: () => { heard.tones.push(frequency.value); },
      stop: () => undefined,
    };
  }

  createGain() {
    return {
      gain: { setValueAtTime: () => undefined, exponentialRampToValueAtTime: () => undefined },
      connect: () => undefined,
    };
  }
}

/** A browser that can play audio, starting quiet; returns what it hears. */
export function browserWithAudio(state: AudioContextState = "running"): HeardAudio {
  heard.tones = [];
  heard.resumed = 0;
  heard.state = state;
  heard.refuse = false;
  vi.stubGlobal("AudioContext", StandInAudioContext);
  return heard;
}
