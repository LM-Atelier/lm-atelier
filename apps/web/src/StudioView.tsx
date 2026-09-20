import { Download, Star, X } from "lucide-react";
import { useCallback, useEffect, useId, useMemo, useReducer, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { GenerationIdentitySummary } from "./GenerationIdentitySummary";
import { StudioOpenImage } from "./StudioOpenImage";
import { ErrorCallout } from "./ErrorCallout";
import { StudioCanvas } from "./StudioCanvas";
import { StudioExtendHandles } from "./StudioExtendHandles";
import { StudioRecipes } from "./StudioRecipes";
import { StudioSelectionTool } from "./StudioSelectionTool";
import { StudioToolGuidance } from "./StudioToolGuidance";
import { StudioToolOptions } from "./StudioToolOptions";
import { StudioToolRail } from "./StudioToolRail";
import { StudioWorkflowSelector } from "./StudioWorkflowSelector";
import { artifactSource } from "./messageMedia";
import { cloneMask, coverage, encodeMaskPng, feather, isEmpty, type MaskRaster } from "./studioMasks";
import { studioApplyPlan } from "./studioApplyPlan";
import { renderLightMap } from "./studioLightMap";
import { readSourcePixels } from "./studioSourcePixels";
import {
  initialToolState,
  snapshotBeforeGesture,
  studioToolReducer,
  toolFor,
  toolUsesMask,
  type StudioToolKind,
} from "./studioToolState";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession, type StudioStep } from "./useStudioSession";
import { useConfirm } from "./useConfirm";
import type { EditTemplate, GenerationIdentity } from "./types";

const SELECTION_NOT_PREPARED =
  "The selection could not be prepared, so nothing was sent. Try again, or clear the selection to edit the whole picture.";
const LIGHT_MAP_NOT_PREPARED =
  "The light map could not be drawn in this browser, so nothing was sent.";

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
  onOpenWorkflows,
  onClose,
}: {
  sourceArtifactId: string | null;
  sourceChatId?: string | null;
  onOpenArtifact: (artifactId: string) => void;
  /** Where a tool that needs an uninstalled workflow sends you. */
  onOpenWorkflows: () => void;
  /** Put the picture down and go back to an empty studio. */
  onClose: () => void;
}) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    heading.current?.focus();
  }, [sourceArtifactId]);
  const { sessionId, steps, previewArtifactId, busy, error, apply } = useStudioSession(
    sourceArtifactId,
    sourceChatId,
  );
  const [confirmDialog, confirm] = useConfirm();
  // Every result is already an artifact in the library - the studio's turns
  // are ordinary turns. What was missing is a way to say "keep this one",
  // because a picture among hundreds is findable only in principle.
  //
  // Read from the artifact rather than remembered locally. A local flag knew
  // only what this visit had done: reopening a picture already marked - from
  // here or from the library - showed it as unmarked, and the control could
  // only ever mark, never take it back.
  const client = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [instruction, setInstruction] = useState("");
  const [selectionError, setSelectionError] = useState<string | null>(null);
  // The recipe an apply should run under. Cleared whenever the instruction is
  // edited by hand: at that point the words are no longer the recipe's, and
  // running its workflow would attribute a result to something it did not do.
  const [recipe, setRecipe] = useState<EditTemplate | null>(null);
  const [workflowAvailability, setWorkflowAvailability] = useState<{
    chatId: string;
    reason: string | null;
  } | null>(null);
  const workflowSelectorId = useId();
  const workflowUnavailable = sessionId && workflowAvailability?.chatId === sessionId
    ? workflowAvailability.reason
    : "Loading the current workflow choice.";
  const recordWorkflowAvailability = useCallback((reason: string | null) => {
    if (!sessionId) return;
    setWorkflowAvailability((currentAvailability) => (
      currentAvailability?.chatId === sessionId
        && currentAvailability.reason === reason
        ? currentAvailability
        : { chatId: sessionId, reason }
    ));
  }, [sessionId]);
  const [tools, dispatch] = useReducer(studioToolReducer, undefined, initialToolState);
  const selectionCoverage = tools.mask ? coverage(tools.mask) : 0;
  // Asked once per visit rather than per apply: installing a workflow is not
  // something that happens while a picture is open.
  const capabilities = useQuery({
    queryKey: ["studio-capabilities"],
    queryFn: api.studioCapabilities,
    // Asked again on every entry, which is what the line above already
    // claimed. Held for a minute instead, the studio told someone who had
    // just followed its own "Browse workflows" button and installed the
    // workflow that the tool was still not installed.
    refetchOnMount: "always",
  });
  const activeTool = capabilities.data?.tools.find((tool) => tool.kind === tools.kind);
  const unavailable = activeTool && !activeTool.available ? activeTool.reason : null;
  // Derived, never synced: with nothing chosen the studio shows the newest
  // result, so a finished apply lands on the canvas without an effect.
  const current = steps.find((step) => step.artifactId === selectedId) ?? steps.at(-1) ?? null;

  const currentArtifactId = current?.artifactId ?? null;
  const artifact = useQuery({
    queryKey: ["artifact", currentArtifactId],
    queryFn: () => api.artifact(currentArtifactId!),
    enabled: Boolean(currentArtifactId),
  });
  const isFavorite = artifact.data?.favorite ?? false;
  const keep = useMutation({
    mutationFn: (next: boolean) => api.favoriteArtifact(currentArtifactId!, next),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["artifact", currentArtifactId] });
      // The library is looking at the same picture.
      void client.invalidateQueries({ queryKey: ["artifacts"] });
    },
  });
  const { bitmap, error: imageError, reload } = useStudioImage(currentArtifactId);
  // Read for the wand only, from the picture on the canvas, and again for each new one.
  const readsColors = tools.kind === "wand";
  const sourcePixels = useMemo(() => (readsColors && bitmap ? readSourcePixels(bitmap) : null), [readsColors, bitmap]);
  // The pointer tool is rebuilt whenever the mode or brush changes; each one
  // is a cheap wrapper over the shared raster, never a copy of it.
  const pointerTool = useMemo(
    () => toolFor(tools, sourcePixels),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tools.kind, tools.brushRadius, tools.mask, tools.selectionMode, tools.colorTolerance, sourcePixels],
  );
  useEffect(() => {
    if (bitmap) {
      dispatch({ type: "image-changed", width: bitmap.width, height: bitmap.height });
    }
  }, [bitmap]);
  // Enhance asks for no words: the whole picture is the subject and
  // the size is the whole instruction. Text takes its words from its
  // own fields, and without a box it would change the whole picture.
  const applyDisabled =
    (tools.kind === "extend" && !Object.values(tools.margins).some(Boolean)) ||
    (tools.kind === "text" && (!tools.newWords.trim() || selectionCoverage === 0)) ||
    (!["enhance", "extend", "text", "relight"].includes(tools.kind) && !instruction.trim()) ||
    busy ||
    !current ||
    Boolean(unavailable) ||
    Boolean(
      workflowUnavailable &&
        !recipe?.workflow_revision_id &&
        !(tools.kind === "relight" && activeTool?.workflow_revision_id),
    );
  if (!sourceArtifactId) {
    return (
      <div className="page-view studio-view">
        <header className="page-header"><div><h1 ref={heading} tabIndex={-1}>Image Studio</h1></div></header>
        <StudioOpenImage onOpened={onOpenArtifact} />
      </div>
    );
  }

  return (
    <div className="page-view studio-view">
      <header className="page-header">
        <div><h1 ref={heading} tabIndex={-1}>Image Studio</h1></div>
        <div className="studio-header-actions">
          {current && !previewArtifactId && (
            <>
              {/* Every result is already in the library - the close dialog
                  beside this says so. What this does is mark one, which is
                  what makes it findable among hundreds, and it is named for
                  that now rather than for saving something already saved. */}
              <button
                className="secondary compact-button"
                disabled={keep.isPending || artifact.isLoading}
                aria-pressed={isFavorite}
                onClick={() => keep.mutate(!isFavorite)}
              >
                <Star size={14} aria-hidden="true" fill={isFavorite ? "currentColor" : "none"} />
                {isFavorite ? "Favorited" : "Favorite"}
              </button>
              <a
                className="secondary compact-button"
                href={artifactSource(current.artifactId) ?? undefined}
                download
              >
                <Download size={14} aria-hidden="true" /> Export
              </a>
            </>
          )}
          <button
            className="secondary compact-button"
            disabled={busy}
            onClick={() => {
              // Only the edits are at stake. The source picture is in the
              // library either way, and every result is a durable artifact -
              // what closing loses is the chain that got here, which is the
              // part worth asking about.
              const edited = steps.length > 1;
              if (!edited) {
                onClose();
                return;
              }
              void confirm({
                title: "Close this image?",
                question: `This session has ${steps.length - 1} edit${
                  steps.length === 2 ? "" : "s"
                }. Closing puts the picture down and leaves the chain behind.`,
                detail:
                  "Every result is already in the media library; closing only "
                  + "leaves this chain of edits behind.",
                confirmLabel: "Close it",
              }).then((ok) => ok && onClose());
            }}
          >
            <X size={14} aria-hidden="true" /> Close
          </button>
        </div>
      </header>
      {confirmDialog}
      {/* A save that fails must not look like a save that worked. The button
          only changes on success, so without this the picture silently stays
          unmarked while the label still invites the same press. */}
      {(error || keep.error) && (
        <ErrorCallout message={((error ?? keep.error) as Error).message} />
      )}
      {selectionError && <ErrorCallout message={selectionError} />}
      <div className="studio-layout">
        <StudioToolRail
          active={tools.kind}
          onSelect={(kind: StudioToolKind) => dispatch({ type: "select-tool", kind })}
          onUndo={() => dispatch({ type: "undo" })}
          onRedo={() => dispatch({ type: "redo" })}
          canUndo={tools.history.canUndo}
          canRedo={tools.history.canRedo}
          disabled={!bitmap || Boolean(previewArtifactId)}
          capabilities={capabilities.data?.tools ?? []}
        />
        <div className="studio-stage">
          {previewArtifactId ? (
            <StudioGenerationPreview artifactId={previewArtifactId} />
          ) : bitmap ? (
            <StudioCanvas
              image={bitmap}
              mask={tools.mask}
              tool={pointerTool}
              maskVersion={tools.maskVersion}
              onGestureStart={() => snapshotBeforeGesture(tools)}
              onStrokeEnd={() => dispatch({ type: "stroke-end" })}
            />
          ) : (
            <StudioStageLoading error={imageError} reload={reload} />
          )}
          {!previewArtifactId && bitmap && tools.kind === "extend" && (
            // Over the picture rather than beside it: the frame is the
            // control, so it has to be where the frame is.
            <StudioExtendHandles
              tools={tools}
              dispatch={dispatch}
              size={{ width: bitmap.width, height: bitmap.height }}
            />
          )}
        </div>
        <aside className="studio-panel">
          {sessionId ? (
            <StudioWorkflowSelector
              chatId={sessionId}
              disabled={busy}
              onAvailabilityChange={recordWorkflowAvailability}
              onSelectionChange={() => setRecipe(null)}
            />
          ) : (
            <StudioWorkflowOpening selectorId={workflowSelectorId} />
          )}
          {tools.kind !== "instruct" && tools.kind !== "relight" && (
            <div className="studio-selection-controls">
              <StudioSelectionTool tools={tools} dispatch={dispatch} colorsUnreadable={readsColors && Boolean(bitmap) && !sourcePixels} />
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
          <StudioToolOptions
            tools={tools}
            dispatch={dispatch}
            instruction={instruction}
            onInstructionChange={(value) => {
              setInstruction(value);
              setRecipe(null);
            }}
          />
          <StudioRecipes
            disabled={busy || !current}
            onApply={(chosen) => {
              setRecipe(chosen);
              setInstruction(chosen.instruction);
            }}
          />
          <StudioRecipeWorkflowNotice recipe={recipe} />
          {unavailable && (
            // Beside the button that would fail, and named by the tools that
            // cannot run, so the sentence arrives before the drawing does.
            <StudioToolGuidance reason={unavailable} onOpenWorkflows={onOpenWorkflows} />
          )}
          <button
            className="primary"
            aria-disabled={applyDisabled}
            onClick={() => {
              if (applyDisabled || !current) return;
              const selection = toolUsesMask(tools.kind) && tools.mask && !isEmpty(tools.mask)
                ? tools.mask
                : null;
              const plan = studioApplyPlan(tools, instruction, recipe, activeTool);
              const send = (mask: Blob | null, secondPicture?: Blob) => {
                apply(
                  plan.words,
                  current.artifactId,
                  mask
                    ? {
                        blob: mask,
                        featherPx: tools.featherPx,
                        invert: false,
                        ...(plan.blendSelection ? { apply: "blend" as const } : {}),
                      }
                    : undefined,
                  plan.settings,
                  plan.workflowRevisionId,
                  () => {
                    setInstruction("");
                    setSelectedId(null);
                  },
                  secondPicture,
                );
              };
              setSelectionError(null);
              if (plan.sendsLightMap) {
                // Drawn at the picture's own size, so the map and the picture line up.
                if (!bitmap) return;
                // Drawing can throw as well as come back empty; both refuse the same way.
                void renderLightMap(bitmap.width, bitmap.height, tools.lightDirection).then(
                  (map) => (map ? send(null, map) : setSelectionError(LIGHT_MAP_NOT_PREPARED)),
                  () => setSelectionError(LIGHT_MAP_NOT_PREPARED),
                );
              } else if (selection) {
                // A selection that cannot be encoded is refused, never sent as an
                // edit of the whole picture it was drawn to protect.
                void encodeMaskPng(plan.blendSelection ? softened(selection, tools.featherPx) : selection).then(
                  (mask) => (mask ? send(mask) : setSelectionError(SELECTION_NOT_PREPARED)),
                  () => setSelectionError(SELECTION_NOT_PREPARED),
                );
              } else send(null);
            }}
          >
            {busy
              ? "Applying…"
              : tools.kind === "extend"
                ? "Extend"
                : tools.kind === "text"
                  ? "Replace words"
                : tools.kind === "relight"
                  ? "Relight"
                : tools.kind === "enhance"
                  ? `Enlarge ${tools.upscaleFactor}x`
                : tools.kind !== "instruct" && selectionCoverage > 0
                  ? "Apply to selection"
                  : "Apply edit"}
          </button>
        </aside>
      </div>
      <StudioFilmstrip
        steps={steps} generationIdentity={previewArtifactId ? null : current?.generationIdentity ?? artifact.data?.generation_identity}
        selectedId={previewArtifactId ? null : current?.artifactId ?? null}
        onSelect={setSelectedId}
      />
    </div>
  );
}

/** A copy of the selection with softened edges, leaving the one on the canvas as drawn.
 *
 * Text is placed back through its box, and a hard edge would show wherever the
 * edited picture differs slightly from the source just outside the words.
 */
function softened(mask: MaskRaster, featherPx: number): MaskRaster {
  const copy = cloneMask(mask);
  if (featherPx > 0) feather(copy, featherPx);
  return copy;
}

function StudioGenerationPreview({ artifactId }: { artifactId: string }) {
  return (
    <figure className="studio-generation-preview">
      <img src={artifactSource(artifactId) ?? undefined} alt="Generation preview" />
      <figcaption role="status">Generation preview</figcaption>
    </figure>
  );
}

function StudioStageLoading({
  error,
  reload,
}: {
  error: string | null;
  reload: () => void;
}) {
  if (error) {
    // A picture that cannot be read is not one still arriving, and "Loading
    // the image" forever is the more comfortable of the two.
    return (
      <div className="studio-stage-loading" role="alert">
        <p>{error}</p>
        <button className="secondary compact-button" onClick={reload}>Try again</button>
      </div>
    );
  }
  // Not an empty state: the empty-state tile is styled to say "nothing here",
  // which is the opposite of what is happening.
  return (
    <div className="studio-stage-loading" role="status">
      <div className="loading-line" />
      <p>Loading the image…</p>
    </div>
  );
}

function StudioRecipeWorkflowNotice({ recipe }: { recipe: EditTemplate | null }) {
  if (!recipe?.workflow_revision_id) return null;
  return (
    <small role="status">
      {recipe.name} supplies the workflow for this edit.
    </small>
  );
}

function StudioWorkflowOpening({ selectorId }: { selectorId: string }) {
  return (
    <div className="workflow-selector studio-workflow-selector">
      <label htmlFor={selectorId}>Editing workflow</label>
      <select id={selectorId} disabled value="">
        <option value="">Opening Studio session…</option>
      </select>
    </div>
  );
}

function StudioFilmstrip({
  steps,
  selectedId,
  generationIdentity,
  onSelect,
}: {
  steps: StudioStep[];
  selectedId: string | null;
  generationIdentity?: GenerationIdentity | null;
  onSelect: (artifactId: string) => void;
}) {
  if (steps.length === 0) return null;
  // A group of buttons, not a listbox: a real listbox owns focus with a
  // roving tabindex and aria-activedescendant, and role="option" would
  // override the native button role so these stop announcing as activatable.
  return (
    <>
      <div className="studio-filmstrip" role="group" aria-label="Edit history">
      {steps.map((step, index) => (
        <button
          key={`${step.messageId}-${step.artifactId}`}
          aria-pressed={step.artifactId === selectedId}
          className={step.artifactId === selectedId ? "selected" : ""}
          onClick={() => onSelect(step.artifactId)}
          onFocus={(event) => event.currentTarget.scrollIntoView({ block: "nearest", inline: "nearest" })}
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
      <GenerationIdentitySummary identity={generationIdentity} />
    </>
  );
}
