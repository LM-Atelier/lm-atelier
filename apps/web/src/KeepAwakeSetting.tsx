import { useEffect, useId, useRef, useState } from "react";
import { api } from "./api";
import type { KeepAwakeStatus } from "./types";

// Jobs start and finish while the page is open, so what holds the computer
// awake is read again on this interval rather than only once.
const REFRESH_MS = 5000;

function describe(status: KeepAwakeStatus): string {
  if (!status.supported) {
    return "This computer gives LM Atelier no way to keep it awake, so long work can stop if it sleeps.";
  }
  if (!status.enabled) return "The computer sleeps on its usual schedule, even while work runs.";
  if (status.running_jobs === 0) return "Nothing is running, so the computer sleeps on its usual schedule.";
  if (status.active) {
    return status.running_jobs === 1
      ? "Keeping the computer awake while 1 job runs."
      : `Keeping the computer awake while ${status.running_jobs} jobs run.`;
  }
  return "The computer refused to stay awake. The work goes on, but can stop if the computer sleeps.";
}

export function KeepAwakeSetting() {
  const id = useId();
  const [status, setStatus] = useState<KeepAwakeStatus | null>(null);
  const [failure, setFailure] = useState("");
  const [saving, setSaving] = useState(false);
  // A save answers with the newest status; a read that started before it
  // must not put the old one back. No read starts while a save runs, so the
  // save's answer, or its failure, is always the one shown.
  const sequence = useRef(0);
  const saveRunning = useRef(false);
  useEffect(() => {
    let live = true;
    const read = async () => {
      if (saveRunning.current) return;
      const mine = ++sequence.current;
      try {
        const next = await api.keepAwake();
        if (live && sequence.current === mine) {
          setStatus(next);
          setFailure("");
        }
      } catch {
        if (live && sequence.current === mine) setFailure("Whether work keeps the computer awake could not be read.");
      }
    };
    void read();
    const timer = window.setInterval(() => void read(), REFRESH_MS);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, []);
  const choose = async (enabled: boolean) => {
    if (!status || saveRunning.current || enabled === status.enabled) return;
    ++sequence.current;
    saveRunning.current = true;
    setSaving(true);
    setFailure("");
    try {
      setStatus(await api.updateKeepAwake({ enabled }));
    } catch {
      setFailure("The setting could not be saved, so it was not changed.");
    } finally {
      saveRunning.current = false;
      setSaving(false);
    }
  };
  const on = status?.enabled === true;
  const off = status?.enabled === false;
  return (
    <section>
      <div className="detail-title"><div><h2>Sleep</h2><p>Saved for this installation.</p></div></div>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-awake`}>While work runs</strong>
          <small id={`${id}-awake-help`}>
            Generation, downloads and runtime setup, never chat. The display still turns off, and sleep you choose, closing the lid or a low battery still put the computer to sleep.
          </small>
        </span>
        <div className="segmented" role="group" aria-labelledby={`${id}-awake`} aria-describedby={`${id}-awake-help`}>
          <button type="button" className={off ? "active" : ""} aria-pressed={off}
            aria-disabled={!status || saving} onClick={() => void choose(false)}>
            Let it sleep
          </button>
          <button type="button" className={on ? "active" : ""} aria-pressed={on}
            aria-disabled={!status || saving} onClick={() => void choose(true)}>
            Keep it awake
          </button>
        </div>
      </div>
      {status && <p className="muted">{describe(status)}</p>}
      {failure && <p role="alert">{failure}</p>}
    </section>
  );
}
