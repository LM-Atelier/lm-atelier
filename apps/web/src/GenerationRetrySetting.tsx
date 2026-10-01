import { useEffect, useId, useRef, useState } from "react";
import { api } from "./api";
import type { GenerationRetryPolicy } from "./generationRetryTypes";

export function GenerationRetrySetting() {
  const id = useId();
  const [policy, setPolicy] = useState<GenerationRetryPolicy | null>(null);
  const [draft, setDraft] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [phase, setPhase] = useState<"loading" | "ready" | "saving" | "failed">("loading");
  const [notice, setNotice] = useState("");
  const request = useRef(0);
  const busy = useRef(false);
  useEffect(() => {
    const sequence = ++request.current;
    busy.current = true;
    void (async () => {
      try {
        const saved = await api.generationRetryPolicy();
        if (request.current !== sequence) return;
        setPolicy(saved);
        setDraft(String(saved.max_retries));
        setPhase("ready");
      } catch {
        if (request.current === sequence) {
          setPhase("failed");
          setNotice("Generation retry settings could not be loaded. Refresh and try again.");
        }
      } finally {
        if (request.current === sequence) busy.current = false;
      }
    })();
    return () => { request.current += 1; };
  }, [refresh]);
  const count = /^\d+$/.test(draft) ? Number(draft) : NaN;
  const valid = Number.isSafeInteger(count) && count >= 0 && count <= 10;
  const save = async () => {
    if (!policy || !valid || busy.current || phase !== "ready") return;
    const sequence = ++request.current;
    busy.current = true;
    setPhase("saving");
    setNotice("");
    try {
      const saved = await api.updateGenerationRetryPolicy(count, policy.revision);
      if (request.current !== sequence) return;
      setPolicy(saved);
      setDraft(String(saved.max_retries));
      setPhase("ready");
      setNotice("Automatic retry setting saved.");
    } catch {
      if (request.current === sequence) {
        setPhase("failed");
        setNotice("The retry setting could not be saved. Refresh and try again.");
      }
    } finally {
      if (request.current === sequence) busy.current = false;
    }
  };
  return <section>
    <div className="detail-title"><div><h2>Generation retries</h2><p>Saved for this workspace.</p></div></div>
    <div className="setting-row appearance-row">
      <span>
        <label htmlFor={`${id}-count`}><strong>Automatic retries</strong></label>
        <small id={`${id}-help`}>Images and videos only. Retry a failed generation up to this many additional times. Cancelled work never retries. Set 0 to turn this off.</small>
      </span>
      <div>
        <input id={`${id}-count`} type="number" min={0} max={10} step={1} value={draft}
          aria-describedby={`${id}-help`} readOnly={phase !== "ready"}
          onChange={(event) => setDraft(event.target.value)} />
        <button type="button" className="secondary" aria-disabled={phase !== "ready" || !valid || count === policy?.max_retries}
          onClick={() => { if (count !== policy?.max_retries) void save(); }}>
          {phase === "saving" ? "Saving…" : "Save retries"}
        </button>
      </div>
    </div>
    {phase === "loading" && <p>Loading retry settings…</p>}
    {notice && <p role={phase === "failed" ? "alert" : "status"}>{notice}</p>}
    {phase === "failed" && <button type="button" className="secondary" onClick={() => {
      if (busy.current) return;
      setPhase("loading"); setNotice(""); setRefresh((value) => value + 1);
    }}>Refresh retry settings</button>}
  </section>;
}
