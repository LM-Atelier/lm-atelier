import { useState } from "react";
import type { SourceFitMode } from "./sourceFit";
import { SourceFitPreview } from "./SourceFitPreview";
import { sourceCanvasDimension, type useSourceFitCanvas } from "./useSourceFitCanvas";

const MODE_LABELS: Record<SourceFitMode, string> = { extend: "Extend / preserve all", crop: "Crop / fill" };

function CanvasDimension({ label, value, onChange, onEdit }: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  onEdit: () => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <label className="setting-row" style={{ gridTemplateColumns: "1fr", gap: 4, padding: "8px 0", borderBottom: 0 }}>
      {label}
      <input type="number" min={1} max={1_000_000} step={1} value={draft ?? String(value)}
        style={{ minWidth: 0, width: "100%", boxSizing: "border-box" }}
        onChange={(event) => {
          const text = event.target.value;
          setDraft(text);
          onEdit();
          if (text.trim() !== "" && sourceCanvasDimension(Number(text))) onChange(Number(text));
        }}
        onBlur={() => setDraft(null)} />
    </label>
  );
}

export function SourceFitControl({
  canvas, sourceUrl, initialWidth, initialHeight,
}: {
  canvas: ReturnType<typeof useSourceFitCanvas>;
  sourceUrl: string;
  initialWidth: unknown;
  initialHeight: unknown;
}) {
  const intent = canvas.value?.request;
  // Switching between the two keeps the canvas size the person already chose.
  const start = (mode: SourceFitMode) => canvas.choose(intent ? { ...intent, mode } : {
    mode,
    width: sourceCanvasDimension(initialWidth) ? initialWidth : 1024,
    height: sourceCanvasDimension(initialHeight) ? initialHeight : 1024,
  });
  return (
    <fieldset style={{ minWidth: 0, margin: 0, padding: 12, border: "1px solid var(--border)", borderRadius: 8 }}>
      <legend>Source canvas · this turn</legend>
      <div className="segmented compact" role="group" aria-label="Source canvas mode" style={{ flexWrap: "wrap" }}>
        <button type="button" aria-pressed={!intent} className={!intent ? "active" : ""} onClick={() => canvas.choose(null)}>Workflow size</button>
        {canvas.modes.map((mode) => (
          <button key={mode} type="button" aria-pressed={intent?.mode === mode} className={intent?.mode === mode ? "active" : ""}
            onClick={() => start(mode)}>{MODE_LABELS[mode]}</button>
        ))}
      </div>
      {!canvas.available && <p className="muted">
        {canvas.missingSource ? "Attach the image you want to fit to a canvas."
          : canvas.missingRevision ? "Choose an image workflow to preview a source canvas."
          : canvas.checking ? "Checking source canvas options…"
          : canvas.capabilityError ? "Cannot check this workflow right now."
          : "This workflow does not offer a source canvas."}
      </p>}
      {intent && <>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 10rem), 1fr))", gap: 8 }}>
          <CanvasDimension label="Canvas width" value={intent.width} onEdit={canvas.invalidatePreview}
            onChange={(width) => canvas.choose({ ...intent, width })} />
          <CanvasDimension label="Canvas height" value={intent.height} onEdit={canvas.invalidatePreview}
            onChange={(height) => canvas.choose({ ...intent, height })} />
        </div>
        <button type="button" className="primary" style={{ marginBlock: 8 }} aria-disabled={!canvas.canPreview} onClick={canvas.requestPreview}>
          Preview canvas
        </button>
        {canvas.pending && <p role="status">Checking source canvas…</p>}
        {canvas.error && <p role="alert">This source and workflow cannot use that canvas. Change the dimensions or choose another workflow.</p>}
        {canvas.preview && <SourceFitPreview preview={canvas.preview} sourceUrl={sourceUrl} />}
      </>}
    </fieldset>
  );
}
