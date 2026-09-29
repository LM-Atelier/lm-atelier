import type { Dispatch } from "react";
import { StudioAdjustTool } from "./StudioAdjustTool";
import { StudioBlurTool } from "./StudioBlurTool";
import { StudioCanvasSizeTool } from "./StudioCanvasSizeTool";
import { StudioCaptionTool } from "./StudioCaptionTool";
import { StudioPaintTool } from "./StudioPaintTool";
import { StudioPerspectiveTool } from "./StudioPerspectiveTool";
import { StudioCropTool } from "./StudioCropTool";
import { StudioResizeTool } from "./StudioResizeTool";
import { StudioTransformTool } from "./StudioTransformTool";
import { pictureCorners } from "./studioPerspective";
import type { StudioToolAction, StudioToolState } from "./studioToolState";
import type { StudioLocalEditDetails, StudioLocalEditOperation } from "./types";

/** The panel's tool-specific control: what this tool needs said before it runs.
 *
 * Extend and Enhance are a drag or a number, Text is the words before and
 * after, Isolate needs nothing, replacing a subject needs the picture it comes
 * from, turning and flipping are a press each, a perspective correction is
 * four corners placed on the picture, a crop is a drawn box, a
 * resize is a width and a height, a canvas change is those and a place, light
 * and color are four sliders, a blur is a marked area and a strength, a paint
 * is a marked area and a color, added words are the words and their look, and
 * every other tool is described in words.
 * Kept apart from the studio view so each tool's control reads in one place.
 */
export function StudioToolOptions({
  tools,
  dispatch,
  instruction,
  onInstructionChange,
  onLocalEdit,
  busy = false,
}: {
  tools: StudioToolState;
  dispatch: Dispatch<StudioToolAction>;
  instruction: string;
  onInstructionChange: (value: string) => void;
  /** Makes an edit that needs no model, such as a turn or a flip. */
  onLocalEdit?: (operation: StudioLocalEditOperation, details?: StudioLocalEditDetails) => void;
  busy?: boolean;
}) {
  if (tools.kind === "transform") {
    return onLocalEdit ? (
      <StudioTransformTool busy={busy} onEdit={onLocalEdit} degrees={tools.straightenDegrees}
        onDegrees={(degrees) => dispatch({ type: "set-straighten", degrees })}
        onStraighten={() => onLocalEdit("straighten", { straighten: { degrees: tools.straightenDegrees } })} />
    ) : null;
  }
  if (tools.kind === "perspective") {
    const size = tools.mask ? { width: tools.mask.width, height: tools.mask.height } : null;
    return onLocalEdit && size ? (
      <StudioPerspectiveTool corners={tools.perspective ?? pictureCorners(size.width, size.height)} size={size}
        busy={busy} onReset={() => dispatch({ type: "set-perspective", corners: null })}
        onApply={(corners) => onLocalEdit("perspective", { perspective: corners })} />
    ) : null;
  }
  if (tools.kind === "crop") {
    return onLocalEdit ? (
      <StudioCropTool mask={tools.mask} maskVersion={tools.maskVersion} shape={tools.cropShape} busy={busy}
        onShape={(shape) => dispatch({ type: "set-crop-shape", shape })}
        onCrop={(box) => onLocalEdit("crop", { crop: box })} />
    ) : null;
  }
  if (tools.kind === "blur") {
    return onLocalEdit ? (
      <StudioBlurTool mask={tools.mask} maskVersion={tools.maskVersion} featherPx={tools.featherPx}
        style={tools.blurStyle} radius={tools.blurRadius} block={tools.pixelBlock} busy={busy}
        onStyle={(style) => dispatch({ type: "set-blur-style", style })}
        onRadius={(radius) => dispatch({ type: "set-blur-radius", radius })}
        onBlock={(block) => dispatch({ type: "set-pixel-block", block })}
        onBlur={(selection) => onLocalEdit("blur", { blur: { selection, radius: tools.blurRadius } })}
        onPixelate={(selection) => onLocalEdit("pixelate", { pixelate: { selection, block: tools.pixelBlock } })} />
    ) : null;
  }
  if (tools.kind === "caption") {
    const size = tools.mask ? { width: tools.mask.width, height: tools.mask.height } : null;
    return onLocalEdit ? (
      <StudioCaptionTool caption={tools.caption} size={size} busy={busy}
        onChange={(patch) => dispatch({ type: "set-caption", patch })}
        onAdd={(words) => onLocalEdit("caption", { caption: { words } })} />
    ) : null;
  }
  if (tools.kind === "paint") {
    return onLocalEdit ? (
      <StudioPaintTool mask={tools.mask} maskVersion={tools.maskVersion} featherPx={tools.featherPx}
        color={tools.paintColor} opacity={tools.paintOpacity} busy={busy}
        onColor={(color) => dispatch({ type: "set-paint-color", color })}
        onOpacity={(opacity) => dispatch({ type: "set-paint-opacity", opacity })}
        onPaint={(selection) => onLocalEdit("paint", {
          paint: { selection, color: tools.paintColor, opacity: tools.paintOpacity },
        })} />
    ) : null;
  }
  if (tools.kind === "adjust") {
    return onLocalEdit ? (
      <StudioAdjustTool adjustments={tools.adjustments} busy={busy}
        onChange={(key, value) => dispatch({ type: "set-adjustment", key, value })}
        onReset={() => dispatch({ type: "reset-adjustments" })}
        onApply={() => onLocalEdit("adjust", { adjustments: tools.adjustments })} />
    ) : null;
  }
  if (tools.kind === "canvas") {
    const size = tools.mask ? { width: tools.mask.width, height: tools.mask.height } : null;
    return onLocalEdit && size ? (
      <StudioCanvasSizeTool key={`${size.width}x${size.height}`} size={size} busy={busy}
        onChange={(canvas) => onLocalEdit("canvas", { canvas })} />
    ) : null;
  }
  if (tools.kind === "resize") {
    // The selection raster is made at the picture's own size, so it says what that is.
    const size = tools.mask ? { width: tools.mask.width, height: tools.mask.height } : null;
    return onLocalEdit && size ? (
      <StudioResizeTool key={`${size.width}x${size.height}`} size={size} busy={busy}
        onResize={(next) => onLocalEdit("resize", { size: next })} />
    ) : null;
  }
  if (tools.kind === "extend") {
    return (
      <div className="studio-tool-options">
        <span>
          <strong>Extend by</strong>
        </span>
        <small>
          {Object.values(tools.margins).some(Boolean)
            ? (["top", "right", "bottom", "left"] as const)
                .filter((side) => tools.margins[side] > 0)
                .map((side) => `${side} ${Math.round(tools.margins[side] * 100)}%`)
                .join(", ")
            : "Drag an edge of the picture outward, or use the arrow keys on one."}
        </small>
        <button
          className="secondary compact-button"
          onClick={() => dispatch({ type: "clear-margins" })}
        >
          Reset edges
        </button>
      </div>
    );
  }
  if (tools.kind === "enhance") {
    return (
      <label>
        <span>
          <strong>Enlarge by</strong> {tools.upscaleFactor}x
        </span>
        <input
          type="range"
          min={1}
          max={8}
          step={1}
          value={tools.upscaleFactor}
          onChange={(event) =>
            dispatch({ type: "set-upscale-factor", factor: Number(event.target.value) })
          }
        />
      </label>
    );
  }
  if (tools.kind === "relight") {
    return (
      <div className="studio-tool-options">
        <div className="segmented" role="group" aria-label="Light from">
          {(["left", "top", "right"] as const).map((direction) => (
            <button
              key={direction}
              type="button"
              aria-pressed={tools.lightDirection === direction}
              className={tools.lightDirection === direction ? "active" : ""}
              onClick={() => dispatch({ type: "set-light-direction", direction })}
            >
              {direction === "left" ? "Left" : direction === "top" ? "Top" : "Right"}
            </button>
          ))}
        </div>
        <label>
          <span>
            <strong>Strength</strong> {Math.round(tools.lightIntensity * 100)}%
          </span>
          <input
            type="range"
            min={25}
            max={100}
            step={5}
            value={Math.round(tools.lightIntensity * 100)}
            onChange={(event) =>
              dispatch({ type: "set-light-intensity", intensity: Number(event.target.value) / 100 })
            }
          />
        </label>
        <div className="segmented" role="group" aria-label="Warmth">
          {([["Neutral", null], ["Warm", 4500], ["Cool", 7500]] as const).map(([label, kelvin]) => (
            <button
              key={label}
              type="button"
              aria-pressed={tools.lightKelvin === kelvin}
              className={tools.lightKelvin === kelvin ? "active" : ""}
              onClick={() => dispatch({ type: "set-light-kelvin", kelvin })}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    );
  }
  if (tools.kind === "isolate") {
    return (
      <div className="studio-tool-options">
        <small>
          Keeps the subject and makes everything behind it transparent. There is nothing to
          select or describe: the workflow finds the subject itself.
        </small>
      </div>
    );
  }
  if (tools.kind === "background") {
    return (
      <div className="studio-tool-options">
        <small>
          Cuts the subject out first, then redraws everything around it. The subject
          keeps its own pixels.
        </small>
        <label>
          <span>
            <strong>Describe the new background</strong>
          </span>
          <textarea
            rows={4}
            value={instruction}
            placeholder="e.g. a quiet beach at sunset"
            onChange={(event) => onInstructionChange(event.target.value)}
          />
        </label>
      </div>
    );
  }
  if (tools.kind === "subject") {
    return (
      <div className="studio-tool-options">
        <small>
          Cuts the subject out first, then redraws it from a second picture. Everything
          around it keeps its own pixels.
        </small>
        <label>
          <span>
            <strong>Picture of the new subject</strong>
          </span>
          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              // A dialog closed without a choice keeps the picture already chosen.
              const picture = event.target.files?.[0];
              if (picture) dispatch({ type: "set-subject-picture", picture });
            }}
          />
        </label>
        {tools.subjectPicture && <small>{tools.subjectPicture.name}</small>}
        <label>
          <span>
            <strong>What to take from it</strong> (optional)
          </span>
          <input
            type="text"
            value={instruction}
            placeholder="e.g. the dog"
            onChange={(event) => onInstructionChange(event.target.value)}
          />
        </label>
      </div>
    );
  }
  if (tools.kind === "text") {
    return (
      <div className="studio-tool-options">
        <small>
          Draw a box around the words. Only what is inside the box changes.
        </small>
        <label>
          <span>
            <strong>Words there now</strong> (optional)
          </span>
          <input
            type="text"
            value={tools.currentWords}
            onChange={(event) => dispatch({ type: "set-current-words", words: event.target.value })}
          />
        </label>
        <label>
          <span>
            <strong>Replace with</strong>
          </span>
          <input
            type="text"
            value={tools.newWords}
            onChange={(event) => dispatch({ type: "set-new-words", words: event.target.value })}
          />
        </label>
      </div>
    );
  }
  return (
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
        onChange={(event) => onInstructionChange(event.target.value)}
      />
    </label>
  );
}
