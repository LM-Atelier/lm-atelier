import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { api } from "./api";
import type { WorkspaceLockPolicy, WorkspaceLockPolicyWrite } from "./workspaceLockTypes";
import { applyWorkspaceLockStatus, workspaceLockFailure, workspaceLockGeneration } from "./workspaceLockState";

type PinTask = "set" | "change" | "remove" | "turn-off" | "lock-later";

const TASKS: Record<PinTask, { title: string; action: string; done: string; asksCurrent: boolean; asksNew: boolean }> = {
  set: { title: "Set a PIN", action: "Save PIN", done: "PIN saved. Unlocking now asks for it.", asksCurrent: false, asksNew: true },
  change: { title: "Change the PIN", action: "Change PIN", done: "PIN changed.", asksCurrent: true, asksNew: true },
  remove: { title: "Remove the PIN", action: "Remove PIN", done: "PIN removed. Anyone at this computer can unlock.", asksCurrent: true, asksNew: false },
  "turn-off": { title: "Turn the lock off", action: "Turn off", done: "Workspace lock turned off.", asksCurrent: true, asksNew: false },
  "lock-later": { title: "Lock later when unused", action: "Save", done: "Lock when unused changed.", asksCurrent: true, asksNew: false },
};

/** The quiet spells offered, in minutes; null never locks for being unused. */
const IDLE_CHOICES: readonly (number | null)[] = [null, 5, 15, 30, 60];

function idleLabel(minutes: number | null): string {
  if (minutes === null) return "Never";
  return minutes === 60 ? "1 hour" : `${minutes} min`;
}

/** Read the lock again so this window follows a setting it just changed. */
async function followSavedLock(): Promise<void> {
  const sentAt = workspaceLockGeneration();
  try {
    applyWorkspaceLockStatus(await api.workspaceLockStatus(), sentAt);
  } catch {
    // The next status read, or the next refusal, brings the window up to date.
  }
}

const SETTING_UNREADABLE = "The saved lock setting could not be read. The privacy documentation explains how to reset it.";

/** The words for a refused change. None of them repeats what was typed. */
function refusalText(error: unknown): string {
  const refusal = workspaceLockFailure(error);
  const wait = refusal.retryAfterSeconds > 0
    ? ` Try again in ${refusal.retryAfterSeconds} ${refusal.retryAfterSeconds === 1 ? "second" : "seconds"}.`
    : "";
  switch (refusal.code) {
    case "workspace-pin-refused": return `The current PIN was not accepted.${wait}`;
    case "workspace-pin-throttled": return `Too many attempts.${wait || " Try again shortly."}`;
    case "workspace-pin-invalid":
    case "request-validation-invalid": return "A PIN needs 4 to 64 characters.";
    case "workspace-pin-unavailable": return "The PIN could not be checked just now. Try again.";
    case "workspace-lock-setting-invalid": return SETTING_UNREADABLE;
    default: return "The lock setting could not be saved. Try again.";
  }
}

function characters(value: string): number {
  return [...value].length;
}

/** Turning the workspace lock on and off, its PIN, and locking straight away.
 *
 * Saved for the whole workspace rather than this browser, because the server
 * is what enforces it. Each change names the revision it was read at, so a
 * change made in another window is never overwritten unseen. PIN fields are
 * emptied after every attempt, and nothing typed in them is shown back.
 */
export function WorkspaceLockSettings() {
  const id = useId();
  const [policy, setPolicy] = useState<WorkspaceLockPolicy | null>(null);
  const [phase, setPhase] = useState<"loading" | "ready" | "saving" | "failed">("loading");
  const [notice, setNotice] = useState<{ text: string; alert: boolean } | null>(null);
  const [task, setTask] = useState<PinTask | null>(null);
  const [currentPin, setCurrentPin] = useState("");
  const [newPin, setNewPin] = useState("");
  const [confirmPin, setConfirmPin] = useState("");
  const [laterIdle, setLaterIdle] = useState<number | null>(null);
  const [refresh, setRefresh] = useState(0);
  const request = useRef(0);
  const busy = useRef(false);
  const firstField = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const sequence = ++request.current;
    busy.current = true;
    void (async () => {
      try {
        const saved = await api.workspaceLockPolicy();
        if (request.current !== sequence) return;
        setPolicy(saved);
        setPhase("ready");
      } catch (error) {
        if (request.current !== sequence) return;
        setPhase("failed");
        setNotice({
          alert: true,
          text: workspaceLockFailure(error).code === "workspace-lock-setting-invalid"
            ? SETTING_UNREADABLE
            : "The lock setting could not be loaded. Refresh and try again.",
        });
      } finally {
        if (request.current === sequence) busy.current = false;
      }
    })();
    return () => { request.current += 1; };
  }, [refresh]);

  useEffect(() => {
    if (task) firstField.current?.focus();
  }, [task]);

  const clearPins = () => {
    setCurrentPin("");
    setNewPin("");
    setConfirmPin("");
  };

  const save = async (write: WorkspaceLockPolicyWrite, done: string) => {
    if (busy.current) return;
    const sequence = ++request.current;
    busy.current = true;
    setPhase("saving");
    setNotice(null);
    try {
      const saved = await api.updateWorkspaceLockPolicy(write);
      if (request.current !== sequence) return;
      setPolicy(saved);
      setTask(null);
      setPhase("ready");
      setNotice({ alert: false, text: done });
      void followSavedLock();
    } catch (error) {
      if (request.current !== sequence) return;
      if (workspaceLockFailure(error).code === "workspace-lock-setting-stale") {
        setTask(null);
        setPhase("loading");
        setNotice({ alert: true, text: "The lock setting was changed somewhere else. Check it below and try again." });
        setRefresh((value) => value + 1);
        return;
      }
      setPhase("ready");
      setNotice({ alert: true, text: refusalText(error) });
    } finally {
      if (request.current === sequence) busy.current = false;
    }
  };

  const ready = phase === "ready" && policy !== null;
  const choose = (enabled: boolean) => {
    if (phase !== "ready" || policy === null || enabled === policy.enabled) return;
    if (!enabled && policy.require_pin) {
      clearPins();
      setNotice(null);
      setTask("turn-off");
      return;
    }
    void save({ expected_revision: policy.revision, enabled }, enabled ? "Workspace lock turned on." : "Workspace lock turned off.");
  };

  const chooseIdle = (minutes: number | null) => {
    if (phase !== "ready" || policy === null || !policy.enabled || minutes === (policy.idle_lock_minutes ?? null)) return;
    const saved = policy.idle_lock_minutes ?? null;
    // Staying open longer weakens the lock, so with a PIN it asks for the PIN.
    if (policy.require_pin && saved !== null && (minutes === null || minutes > saved)) {
      clearPins();
      setNotice(null);
      setLaterIdle(minutes);
      setTask("lock-later");
      return;
    }
    void save({ expected_revision: policy.revision, enabled: policy.enabled, idle_lock_minutes: minutes }, "Lock when unused changed.");
  };

  const open = (next: PinTask) => {
    if (phase !== "ready" || policy === null) return;
    clearPins();
    setNotice(null);
    setTask(next);
  };

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (phase !== "ready" || policy === null || task === null) return;
    const shape = TASKS[task];
    const current = currentPin;
    const next = newPin;
    const confirmed = confirmPin;
    clearPins();
    let problem = "";
    if (shape.asksCurrent && !current) problem = "Enter the current PIN.";
    else if (shape.asksNew && (characters(next) < 4 || characters(next) > 64)) problem = "A PIN needs 4 to 64 characters.";
    else if (shape.asksNew && next !== confirmed) problem = "The two PINs do not match.";
    if (problem) {
      setNotice({ alert: true, text: problem });
      firstField.current?.focus();
      return;
    }
    const write: WorkspaceLockPolicyWrite = {
      expected_revision: policy.revision,
      enabled: task === "turn-off" ? false : policy.enabled,
      ...(shape.asksCurrent ? { current_pin: current } : {}),
      ...(shape.asksNew ? { new_pin: next } : {}),
      ...(task === "remove" ? { clear_pin: true } : {}),
      ...(task === "lock-later" ? { idle_lock_minutes: laterIdle } : {}),
    };
    void save(write, shape.done);
  };

  const lockNow = async () => {
    if (phase !== "ready" || policy === null || !policy.enabled || busy.current) return;
    busy.current = true;
    setNotice(null);
    try {
      // Taking the answer locks this window at once, without waiting for the
      // server to close its live connection.
      applyWorkspaceLockStatus(await api.lockWorkspace());
    } catch (error) {
      setNotice({
        alert: true,
        text: workspaceLockFailure(error).code === "workspace-lock-disabled"
          ? "Turn the lock on first."
          : "LM Atelier could not be locked. Try again.",
      });
    } finally {
      busy.current = false;
    }
  };

  const enabled = policy?.enabled ?? false;
  const shape = task ? TASKS[task] : null;
  return (
    <section>
      <div className="detail-title"><div><h2>Workspace lock</h2><p>Saved for this workspace, and applied in every window.</p></div></div>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-lock`}>Lock</strong>
          <small id={`${id}-lock-help`}>
            When on, LM Atelier can be locked from here and starts locked whenever the service starts. Work that is already running keeps going while it is locked.
          </small>
        </span>
        <div className="segmented workspace-lock-switch" role="group" aria-labelledby={`${id}-lock`} aria-describedby={`${id}-lock-help`}>
          <button type="button" className={enabled ? "" : "active"} aria-pressed={policy !== null && !enabled} aria-disabled={!ready} onClick={() => choose(false)}>
            Off
          </button>
          <button type="button" className={enabled ? "active" : ""} aria-pressed={policy !== null && enabled} aria-disabled={!ready} onClick={() => choose(true)}>
            On
          </button>
        </div>
      </div>
      <div className="setting-row appearance-row">
        <span>
          <strong>PIN</strong>
          <small>
            {policy?.require_pin
              ? "Unlocking asks for the PIN."
              : "No PIN is set, so anyone at this computer can unlock."}
          </small>
        </span>
        <div className="workspace-lock-actions">
          {policy?.require_pin ? (<>
            <button type="button" className="secondary" aria-disabled={!ready} onClick={() => open("change")}>Change PIN</button>
            <button type="button" className="secondary" aria-disabled={!ready} onClick={() => open("remove")}>Remove PIN</button>
          </>) : (
            <button type="button" className="secondary" aria-disabled={!ready} onClick={() => open("set")}>Set PIN</button>
          )}
        </div>
      </div>
      {shape && (
        <form className="workspace-lock-form" aria-label={shape.title} onSubmit={submit}>
          {shape.asksCurrent && (
            <label>
              Current PIN
              <input ref={firstField} type="password" autoComplete="off" value={currentPin} onChange={(event) => setCurrentPin(event.target.value)} />
            </label>
          )}
          {shape.asksNew && (<>
            <label>
              New PIN
              <input ref={shape.asksCurrent ? undefined : firstField} type="password" autoComplete="off" value={newPin}
                aria-describedby={`${id}-pin-help`} onChange={(event) => setNewPin(event.target.value)} />
            </label>
            <label>
              Confirm new PIN
              <input type="password" autoComplete="off" value={confirmPin} onChange={(event) => setConfirmPin(event.target.value)} />
            </label>
            <small id={`${id}-pin-help`}>4 to 64 characters. Any characters count, spaces included.</small>
          </>)}
          <div className="workspace-lock-actions">
            <button type="submit" className="primary" aria-disabled={!ready}>{phase === "saving" ? "Saving…" : shape.action}</button>
            <button type="button" className="secondary" onClick={() => { clearPins(); setTask(null); }}>Cancel</button>
          </div>
        </form>
      )}
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-idle`}>Lock when unused</strong>
          <small id={`${id}-idle-help`}>
            {enabled
              ? "Locks once no window has had a key press, click or touch for this long. Running work does not count as use."
              : "Turn the lock on to choose this."}
          </small>
        </span>
        <div className="segmented workspace-lock-switch" role="group" aria-labelledby={`${id}-idle`} aria-describedby={`${id}-idle-help`}>
          {IDLE_CHOICES.map((minutes) => (
            <button
              type="button"
              key={minutes ?? "never"}
              className={policy !== null && (policy.idle_lock_minutes ?? null) === minutes ? "active" : ""}
              aria-pressed={policy !== null && (policy.idle_lock_minutes ?? null) === minutes}
              aria-disabled={!ready || !enabled}
              onClick={() => chooseIdle(minutes)}
            >
              {idleLabel(minutes)}
            </button>
          ))}
        </div>
      </div>
      <div className="setting-row appearance-row">
        <span>
          <strong>Lock now</strong>
          <small id={`${id}-now-help`}>
            {enabled
              ? "Takes the workspace off the screen in every window until it is unlocked."
              : "Turn the lock on to use this."}
          </small>
        </span>
        <div className="workspace-lock-actions">
          <button type="button" className="secondary" aria-disabled={!ready || !enabled} aria-describedby={`${id}-now-help`}
            onClick={() => void lockNow()}>
            Lock now
          </button>
        </div>
      </div>
      {phase === "loading" && <p>Loading the lock setting…</p>}
      {notice && <p role={notice.alert ? "alert" : "status"}>{notice.text}</p>}
      {phase === "failed" && <button type="button" className="secondary" onClick={() => {
        if (busy.current) return;
        setPhase("loading"); setNotice(null); setRefresh((value) => value + 1);
      }}>Refresh lock setting</button>}
    </section>
  );
}
