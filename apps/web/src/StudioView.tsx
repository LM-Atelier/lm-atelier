import { useEffect, useMemo, useReducer, useState } from "react";
import { StudioOpenImage } from "./StudioOpenImage";
import { ErrorCallout } from "./ErrorCallout";
import { StudioCanvas } from "./StudioCanvas";
import { StudioToolRail } from "./StudioToolRail";
import { coverage, encodeMaskPng, isEmpty } from "./studioMasks";
import {
  initialToolState,
  studioToolReducer,
  toolFor,
  type StudioToolKind,
} from "./studioToolState";
import { useStudioSession, type StudioStep } from "./useStudioSession";

/** The Image Studio: a canvas-first editing surface, not a conversation.
 *
 * The center is the current result at zoom; the filmstrip below is the edit
 * chain; the right panel holds the tool's own controls. Applies run as
 * ordinary turns in a hidden session - the user never sees a transcript,
 * only pictures replacing pictures and the instruction that made each.
 */
export function StudioView({
  sourceArtifactId,
  sourceChatId = null,
  onOpenArtifact,
}: {
  sourceArtifactId: string | null;
  sourceChatId?: string | null;
  onOpenArtifact: (artifactId: string) => void;
}) {
  const { steps, busy, error, apply } = useStudioSession(sourceArtifactId, sourceChatId);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [instruction, setInstruction] = useState("");
  const [bitmap, setBitmap] = useState<ImageBitmap | null>(null);
  const [tools, dispatch] = useReducer(studioToolReducer, undefined, initialToolState);
  // The pointer tool is rebuilt whenever the mode or brush changes; each one
  // is a cheap wrapper over the shared raster, never a copy of it.
  const pointerTool = useMemo(
    () => toolFor(tools),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tools.kind, tools.brushRadius, tools.mask],
  );
  const selectionCoverage = tools.mask ? coverage(tools.mask) : 0;

  // Derived, never synced: with nothing chosen the studio shows the newest
  // result, so a finished apply lands on the canvas without an effect.
  const current = steps.find((step) => step.artifactId === selectedId) ?? steps.at(-1) ?? null;

  const currentArtifactId = current?.artifactId ?? null;
  useEffect(() => {
    let live = true;
    if (!currentArtifactId) {
      // Async so the clear never runs synchronously inside the effect.
      void Promise.resolve().then(() => {
        if (live) setBitmap(null);
      });
      return () => {
        live = false;
      };
    }
    void fetch(`/api/artifacts/${encodeURIComponent(currentArtifactId)}/content`)
      .then((response) => response.blob())
      .then((blob) => createImageBitmap(blob))
      .then((decoded) => {
        if (!live) return;
        setBitmap(decoded);
        dispatch({ type: "image-changed", width: decoded.width, height: decoded.height });
      })
      .catch(() => {
        if (live) setBitmap(null);
      });
    return () => {
      live = false;
    };
  }, [currentArtifactId]);

  if (!sourceArtifactId) {
    return (
      <div className="page-view studio-view">
        <header className="page-header"><div><h1>Image Studio</h1></div></header>
        <StudioOpenImage onOpened={onOpenArtifact} />
      </div>
    );
  }

  return (
    <div className="page-view studio-view">
      <header className="page-header">
        <div><h1>Image Studio</h1></div>
      </header>
      {error && <ErrorCallout message={(error as Error).message} />}
      <div className="studio-layout">
        <StudioToolRail
          active={tools.kind}
          onSelect={(kind: StudioToolKind) => dispatch({ type: "select-tool", kind })}
          onUndo={() => dispatch({ type: "undo" })}
          onRedo={() => dispatch({ type: "redo" })}
          canUndo={tools.history.canUndo}
          canRedo={tools.history.canRedo}
          disabled={!bitmap}
        />
        <div className="studio-stage">
          {bitmap ? (
            <StudioCanvas
              image={bitmap}
              mask={tools.mask}
              tool={pointerTool}
              maskVersion={tools.maskVersion}
              onGestureStart={() => dispatch({ type: "gesture-start" })}
              onStrokeEnd={() => dispatch({ type: "stroke-end" })}
            />
          ) : (
            // Not an empty state: the empty-state tile is styled to say
            // "nothing here", which is the opposite of what is happening.
            <div className="studio-stage-loading" role="status">
              <div className="loading-line" />
              <p>Loading the image…</p>
            </div>
          )}
        </div>
        <aside className="studio-panel">
          {tools.kind !== "instruct" && (
            <div className="studio-selection-controls">
              <label>
                Brush size
                <input
                  type="range"
                  min={1}
                  max={200}
                  value={tools.brushRadius}
                  onChange={(event) =>
                    dispatch({ type: "set-brush-radius", radius: Number(event.target.value) })}
                />
              </label>
              <div className="row-actions">
                <button
                  className="secondary compact-button"
                  onClick={() => dispatch({ type: "invert" })}
                >
                  Invert
                </button>
                <button
                  className="secondary compact-button"
                  onClick={() => dispatch({ type: "feather" })}
                >
                  Soften edges
                </button>
                <button
                  className="secondary compact-button"
                  onClick={() => dispatch({ type: "clear" })}
                >
                  Clear
                </button>
              </div>
              <small>
                {selectionCoverage > 0
                  ? `${(selectionCoverage * 100).toFixed(1)}% of the image selected`
                  : "Nothing selected yet - paint over what you want to change."}
              </small>
            </div>
          )}
          <label>
            <span>
              <strong>
                {tools.kind === "instruct" ? "Describe the edit" : "Describe the change here"}
              </strong>
            </span>
            <textarea
              rows={4}
              value={instruction}
              placeholder="e.g. make it a watercolor painting"
              onChange={(event) => setInstruction(event.target.value)}
            />
          </label>
          <button
            className="primary"
            disabled={!instruction.trim() || busy || !current}
            onClick={() => {
              if (!current) return;
              const selection = tools.kind !== "instruct" && tools.mask && !isEmpty(tools.mask)
                ? tools.mask
                : null;
              const send = (mask: Blob | null) => {
                apply(
                  instruction.trim(),
                  current.artifactId,
                  mask ? { blob: mask, featherPx: tools.featherPx, invert: false } : undefined,
                );
                setInstruction("");
                setSelectedId(null);
              };
              if (selection) void encodeMaskPng(selection).then(send);
              else send(null);
            }}
          >
            {busy
              ? "Applying…"
              : tools.kind !== "instruct" && selectionCoverage > 0
                ? "Apply to selection"
                : "Apply edit"}
          </button>
        </aside>
      </div>
      <StudioFilmstrip
        steps={steps}
        selectedId={current?.artifactId ?? null}
        onSelect={setSelectedId}
      />
    </div>
  );
}

function StudioFilmstrip({
  steps,
  selectedId,
  onSelect,
}: {
  steps: StudioStep[];
  selectedId: string | null;
  onSelect: (artifactId: string) => void;
}) {
  if (steps.length === 0) return null;
  return (
    // A group of buttons, not a listbox: a real listbox owns focus with a
    // roving tabindex and aria-activedescendant, and role="option" would
    // override the native button role so these stop announcing as
    // activatable at all.
    <div className="studio-filmstrip" role="group" aria-label="Edit history">
      {steps.map((step, index) => (
        <button
          key={`${step.messageId}-${step.artifactId}`}
          aria-pressed={step.artifactId === selectedId}
          className={step.artifactId === selectedId ? "selected" : ""}
          onClick={() => onSelect(step.artifactId)}
        >
          <img
            src={`/api/artifacts/${encodeURIComponent(step.artifactId)}/content`}
            alt={step.isSource ? "The original image" : `Result of step ${index}`}
            loading="lazy"
          />
          <small>{step.isSource ? "Original" : step.instruction || `Step ${index}`}</small>
        </button>
      ))}
    </div>
  );
}
