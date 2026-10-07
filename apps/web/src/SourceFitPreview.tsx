import { useId } from "react";
import type { SourceFitPreviewResult } from "./sourceFit";

/** Display the server's exact rectangle; the browser does not resolve geometry. */
export function SourceFitPreview({
  preview,
  sourceUrl,
}: {
  preview: SourceFitPreviewResult;
  sourceUrl: string;
}) {
  const id = useId();
  if (preview.mode === "crop" && preview.kept) return <CropPreview preview={preview} kept={preview.kept} sourceUrl={sourceUrl} />;
  const { source, canvas, margins, source_rectangle: rectangle } = preview;
  const added = `Left ${margins.left}, right ${margins.right}, top ${margins.top}, bottom ${margins.bottom} pixels`;
  return (
    <figure style={{ margin: 0, minWidth: 0 }}>
      <svg
        role="img"
        aria-labelledby={`${id}-title ${id}-description`}
        viewBox={`0 0 ${canvas.width} ${canvas.height}`}
        style={{ display: "block", width: "100%", maxHeight: 240 }}
      >
        <title id={`${id}-title`}>Extension preview</title>
        <desc id={`${id}-description`}>
          {`The complete ${source.width} by ${source.height} source sits inside a ${canvas.width} by ${canvas.height} canvas. Added canvas: ${added}.`}
        </desc>
        <rect width={canvas.width} height={canvas.height} fill="var(--accent, #6366f1)" fillOpacity={0.22} />
        <image
          href={sourceUrl}
          x={rectangle.x}
          y={rectangle.y}
          width={rectangle.width}
          height={rectangle.height}
          preserveAspectRatio="xMidYMid meet"
          aria-hidden="true"
        />
        <rect
          x={rectangle.x}
          y={rectangle.y}
          width={rectangle.width}
          height={rectangle.height}
          fill="none"
          stroke="currentColor"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <figcaption>
        <strong>Extend / preserve all</strong>
        <p>{`Source: ${source.width} × ${source.height} · Output: ${canvas.width} × ${canvas.height}`}</p>
        <p className="muted">The shaded area will be generated. The complete source stays visible.</p>
        <p className="muted">{`Added canvas: ${added}.`}</p>
      </figcaption>
    </figure>
  );
}

/** The whole source, with the part the crop keeps outlined and the rest dimmed. */
function CropPreview({
  preview,
  kept,
  sourceUrl,
}: {
  preview: SourceFitPreviewResult;
  kept: NonNullable<SourceFitPreviewResult["kept"]>;
  sourceUrl: string;
}) {
  const id = useId();
  const { source, canvas } = preview;
  const size = (value: number) => Number(value.toFixed(1));
  const outer = `M0 0H${source.width}V${source.height}H0Z`;
  const hole = `M${kept.left} ${kept.top}h${kept.width}v${kept.height}h${-kept.width}Z`;
  return (
    <figure style={{ margin: 0, minWidth: 0 }}>
      <svg
        role="img"
        aria-labelledby={`${id}-title ${id}-description`}
        viewBox={`0 0 ${source.width} ${source.height}`}
        style={{ display: "block", width: "100%", maxHeight: 240 }}
      >
        <title id={`${id}-title`}>Crop preview</title>
        <desc id={`${id}-description`}>
          {`The outlined ${size(kept.width)} by ${size(kept.height)} part of the ${source.width} by ${source.height} source fills the ${canvas.width} by ${canvas.height} canvas. The rest of the source is left out.`}
        </desc>
        <image href={sourceUrl} width={source.width} height={source.height} preserveAspectRatio="none" aria-hidden="true" />
        <path d={`${outer} ${hole}`} fillRule="evenodd" fill="currentColor" fillOpacity={0.45} />
        <rect
          x={kept.left}
          y={kept.top}
          width={kept.width}
          height={kept.height}
          fill="none"
          stroke="currentColor"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <figcaption>
        <strong>Crop / fill</strong>
        <p>{`Source: ${source.width} × ${source.height} · Output: ${canvas.width} × ${canvas.height}`}</p>
        <p className="muted">The outlined part fills the canvas and is edited. The dimmed part is left out.</p>
      </figcaption>
    </figure>
  );
}
