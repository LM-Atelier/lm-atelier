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
