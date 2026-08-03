import { useEffect, useState } from "react";
import { Image as ImageIcon, Wand2 } from "lucide-react";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { StudioCanvas } from "./StudioCanvas";
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
}: {
  sourceArtifactId: string | null;
  sourceChatId?: string | null;
}) {
  const { steps, busy, error, apply } = useStudioSession(sourceArtifactId, sourceChatId);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [instruction, setInstruction] = useState("");
  const [bitmap, setBitmap] = useState<ImageBitmap | null>(null);

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
        if (live) setBitmap(decoded);
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
        <EmptyState
          icon={<Wand2 />}
          title="Open an image to edit"
          body="Choose Edit on any image in the Media library or a chat, and it opens here."
        />
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
        <div className="studio-stage">
          {bitmap ? (
            <StudioCanvas image={bitmap} mask={null} tool={null} />
          ) : (
            <EmptyState icon={<ImageIcon />} title="Loading the image" body="" />
          )}
        </div>
        <aside className="studio-panel">
          <label>
            <span><strong>Describe the edit</strong></span>
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
              apply(instruction.trim(), current.artifactId);
              setInstruction("");
              setSelectedId(null);
            }}
          >
            {busy ? "Applying…" : "Apply edit"}
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
    <div className="studio-filmstrip" role="listbox" aria-label="Edit history">
      {steps.map((step, index) => (
        <button
          key={`${step.messageId}-${step.artifactId}`}
          role="option"
          aria-selected={step.artifactId === selectedId}
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
