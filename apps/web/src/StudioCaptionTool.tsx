import { Captions } from "lucide-react";
import { useState } from "react";
import { StudioAnchorPicker } from "./StudioAnchorPicker";
import { drawCaption, type CaptionFont, type StudioCaption } from "./studioCaption";
import { PAINT_SWATCHES } from "./studioPaint";

const FONTS: Array<{ font: CaptionFont; label: string }> = [
  { font: "sans", label: "Sans" },
  { font: "serif", label: "Serif" },
  { font: "mono", label: "Mono" },
];

/** Adding words to the picture, drawn in the browser and laid down without a model.
 *
 * The words are drawn at the picture's own size in a typeface the app ships,
 * shown on the canvas as they are written, and that same drawing is what is
 * added, so the preview is the result. They sit at one of nine places with a
 * margin from the edges, and an outline in the opposite shade keeps them
 * readable over a busy picture.
 */
export function StudioCaptionTool({
  caption,
  size,
  busy,
  onChange,
  onAdd,
}: {
  caption: StudioCaption;
  /** The picture's size now, in its own pixels; null while it loads. */
  size: { width: number; height: number } | null;
  busy: boolean;
  onChange: (patch: Partial<StudioCaption>) => void;
  /** Receives the drawn words as a transparent PNG at the picture's size. */
  onAdd: (words: Blob) => void;
}) {
  const [preparing, setPreparing] = useState(false);
  const [refusal, setRefusal] = useState<string | null>(null);
  const written = caption.text.trim() !== "";
  const ready = written && size !== null && !busy && !preparing;

  const add = () => {
    if (!ready || !size) return;
    setPreparing(true);
    setRefusal(null);
    const refuse = () => setRefusal("The words could not be drawn. Try again.");
    void drawCaption(size.width, size.height, caption)
      .then((canvas) =>
        canvas
          ? new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"))
          : null,
      )
      .then(
        (blob) => {
          setPreparing(false);
          if (blob) onAdd(blob);
          else refuse();
        },
        () => {
          setPreparing(false);
          refuse();
        },
      );
  };

  return (
    <div className="studio-tool-options">
      <label className="studio-adjust-slider">
        <span>
          <strong>Words</strong>
        </span>
        <textarea
          value={caption.text}
          rows={2}
          aria-label="Words"
          onChange={(event) => onChange({ text: event.target.value })}
        />
      </label>
      <div className="segmented" role="group" aria-label="Typeface">
        {FONTS.map(({ font, label }) => (
          <button
            key={font}
            type="button"
            aria-pressed={caption.font === font}
            className={caption.font === font ? "active" : ""}
            onClick={() => onChange({ font })}
          >
            {label}
          </button>
        ))}
      </div>
      <label className="studio-adjust-slider">
        <span>
          <strong>Size</strong> {caption.sizePercent}% of the height
        </span>
        <input
          type="range"
          min={2}
          max={30}
          step={1}
          value={caption.sizePercent}
          aria-label="Size"
          onChange={(event) => onChange({ sizePercent: Number(event.target.value) })}
        />
      </label>
      <div className="segmented studio-paint-swatches" role="group" aria-label="Text color">
        {PAINT_SWATCHES.map((swatch) => (
          <button
            key={swatch.color}
            type="button"
            aria-label={swatch.label}
            aria-pressed={caption.color === swatch.color}
            className={caption.color === swatch.color ? "active" : ""}
            onClick={() => onChange({ color: swatch.color })}
          >
            <span className="studio-paint-swatch" style={{ backgroundColor: swatch.color }} />
          </button>
        ))}
      </div>
      <label>
        <input type="checkbox" checked={caption.bold} onChange={(event) => onChange({ bold: event.target.checked })} />
        <span>Bold</span>
      </label>
      <label>
        <input type="checkbox" checked={caption.outline}
          onChange={(event) => onChange({ outline: event.target.checked })} />
        <span>Outline</span>
      </label>
      <StudioAnchorPicker value={caption.anchor} label="Where the words sit" onChange={(anchor) => onChange({ anchor })} />
      <small role={refusal ? "alert" : undefined}>
        {refusal ??
          (busy || preparing
            ? "Applying…"
            : written
              ? "The picture shows the words as they will be added."
              : "Write the words to add, then choose where they sit.")}
      </small>
      <button
        type="button"
        className="secondary compact-button"
        // Not disabled: a focused button that becomes disabled drops focus to
        // the page, and a keyboard user would lose their place.
        aria-disabled={!ready}
        onClick={add}
      >
        <Captions size={14} aria-hidden="true" /> Add the words
      </button>
    </div>
  );
}
