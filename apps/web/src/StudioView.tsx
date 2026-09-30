import { Star } from "lucide-react";
import { useCallback, useEffect, useId, useMemo, useReducer, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { GenerationIdentitySummary } from "./GenerationIdentitySummary";
import { StudioOpenImage } from "./StudioOpenImage";
import { ErrorCallout } from "./ErrorCallout";
import { StudioCanvas } from "./StudioCanvas";
import { StudioCloseButton } from "./StudioCloseButton";
import { StudioCompare } from "./StudioCompare";
import { StudioExportLink } from "./StudioExportLink";
import { StudioHideResult } from "./StudioHideResult";
import { hideStudioStep, showStudioSteps, useStudioHiddenSteps } from "./studioHiddenSteps";
import { StudioTryAnother } from "./StudioTryAnother";
import { StudioUseInChat } from "./StudioUseInChat";
import type { StudioPictureForChat } from "./useStudioPictureForChat";
import { StudioExtendHandles } from "./StudioExtendHandles";
import { StudioResultCount } from "./StudioResultCount";
import { StudioRunningEdit } from "./StudioRunningEdit";
import { StudioApplyFailure } from "./StudioApplyFailure";
import { studioApplyProgress } from "./studioApplyProgress";
import { StudioRecipes } from "./StudioRecipes";
import { StudioSelectionControls } from "./StudioSelectionTool";
import { StudioCapabilityCheck, StudioToolGuidance } from "./StudioToolGuidance";
import { StudioToolOptions } from "./StudioToolOptions";
import { StudioToolRail } from "./StudioToolRail";
import { StudioWorkflowSelector } from "./StudioWorkflowSelector";
import { artifactSource } from "./messageMedia";
import { coverage } from "./studioMasks";
import { studioApplyLabel, studioToolReady } from "./studioApplyPlan";
import { applyStudioEdit } from "./studioApplyEdit";
import { studioStepAncestors, studioStepOrigin } from "./studioStepOrigin";
import { studioRecipeSource } from "./studioRecipeSource";
import { readSourcePixels } from "./studioSourcePixels";
import { useAdjustedPreview } from "./useAdjustedPreview";
import { useCaptionPreview } from "./useCaptionPreview";
import { useStraightenPreview } from "./useStraightenPreview";
import { paintRgb } from "./studioPaint";
import {
  initialToolState,
  snapshotBeforeGesture,
  studioToolReducer,
  toolFor,
  toolMarksPicture,
  toolUsesMask,
  type StudioToolKind,
} from "./studioToolState";
import { useStudioCompare } from "./useStudioCompare";
import { SUBJECT_ELSEWHERE, useStudioSubjectSelection } from "./useStudioSubjectSelection";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession, type StudioStep } from "./useStudioSession";
import { useStudioDraft } from "./useStudioDraft";
import { useStudioBackground } from "./useStudioBackground";
import type { EditTemplate, GenerationIdentity } from "./types";

/** Tools that make their edit without a model, each from its own panel, so they have no Apply of the model's. */
const EXACT_EDITS: readonly string[] = [
  "transform", "perspective", "crop", "resize", "canvas", "adjust", "blur", "paint", "caption",
];

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
  onUseInChat,
}: {
  sourceArtifactId: string | null;
  sourceChatId?: string | null;
  onOpenArtifact: (artifactId: string) => void;
  /** Where a tool that needs an uninstalled workflow sends you. */
  onOpenWorkflows: () => void;
  /** Put the picture down and go back to an empty studio. */
  onClose: () => void;
  /** Attach the picture on screen to a chat's next message. */
  onUseInChat?: (picture: StudioPictureForChat) => void;
}) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    heading.current?.focus();
  }, [sourceArtifactId]);
  const { sessionId, session, steps, previewArtifactId, busy: sessionBusy, error, apply, again, localEdit, stop, stopping } = useStudioSession(
    sourceArtifactId,
    sourceChatId,
  );
  const hidden = useStudioHiddenSteps(sessionId);
  // Every result is already an artifact in the library - the studio's turns
  // are ordinary turns. What was missing is a way to say "keep this one",
  // because a picture among hundreds is findable only in principle.
  //
  // Read from the artifact rather than remembered locally. A local flag knew
  // only what this visit had done: reopening a picture already marked - from
  // here or from the library - showed it as unmarked, and the control could
  // only ever mark, never take it back.
  const client = useQueryClient();
  const draft = useStudioDraft(sessionId);
  const [selectedId, setSelectedId] = useState<string | null>(draft.selectedId);
  const [instruction, setInstruction] = useState(draft.instruction);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [results, setResults] = useState(1);
  // Replacing a background or a subject is two applies; the studio stays busy in between.
  const cutoutEdit = useStudioBackground(sessionId, session, apply, setSelectionError);
  const applyProgress = studioApplyProgress(session);
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
  // Every cutout runs on the workflow Isolate is given.
  const isolateTool = capabilities.data?.tools.find((tool) => tool.kind === "isolate");
  const unavailable = activeTool && !activeTool.available ? activeTool.reason : null;
  // With no report yet, or none to be had, a model's tool waits rather than
  // looking ready: the gap would otherwise be found only when the edit failed.
  const unchecked = capabilities.data ? null : capabilities.isError ? "failed" : "checking";
  // Derived, never synced: with nothing chosen the studio shows the newest
  // result not hidden, so a finished apply lands on the canvas without an effect.
  const shown = steps.filter((step) => step.isSource || !hidden.has(step.artifactId));
  const current = shown.find((step) => step.artifactId === selectedId) ?? shown.at(-1) ?? null;

  const currentArtifactId = current?.artifactId ?? null;
  // Finding the subject is a cutout too, whose alpha becomes the selection of the picture it was found in.
  const subjectSearch = useStudioSubjectSelection(sessionId, session, apply, setSelectionError, (artifactId, mask) =>
    artifactId === currentArtifactId ? dispatch({ type: "select-subject", mask }) : setSelectionError(SUBJECT_ELSEWHERE));
  const busy = sessionBusy || cutoutEdit.busy || subjectSearch.busy;
  // Offered where a workflow can cut a subject out, which the report names under Isolate.
  const subjectWorkflow = isolateTool?.available ? isolateTool.workflow_revision_id : null;
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
  const compare = useStudioCompare(current, bitmap, Boolean(previewArtifactId), steps, hidden);
  // Read for the wand and the light and color preview, from the picture on the canvas, and again for each new one.
  const readsColors = tools.kind === "wand" || tools.kind === "adjust";
  const sourcePixels = useMemo(() => (readsColors && bitmap ? readSourcePixels(bitmap) : null), [readsColors, bitmap]);
  const adjustedPreview = useAdjustedPreview(bitmap, sourcePixels, tools.kind === "adjust" ? tools.adjustments : null);
  const captionPreview = useCaptionPreview(bitmap, tools.kind === "caption" ? tools.caption : null);
  const straightenPreview = useStraightenPreview(bitmap, tools.kind === "transform" ? tools.straightenDegrees : null);
  // The pointer tool is rebuilt whenever the mode or brush changes; each one
  // is a cheap wrapper over the shared raster, never a copy of it.
  const pointerTool = useMemo(
    () => toolFor(tools, sourcePixels, (corners) => dispatch({ type: "set-perspective", corners })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tools.kind, tools.brushRadius, tools.cropShape, tools.perspective, tools.mask, tools.selectionMode,
      tools.colorTolerance, sourcePixels],
  );
  useEffect(() => {
    if (!bitmap) return;
    // Back on the picture the Studio was left on: what was drawn there still fits it.
    const kept = draft.take(currentArtifactId);
    dispatch(kept ? { type: "restore", state: kept.tools } : { type: "image-changed", width: bitmap.width, height: bitmap.height });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bitmap]);
  useEffect(() => draft.track({ artifactId: currentArtifactId, tools, instruction, selectedId }));
  const applyDisabled = unchecked !== null || busy || !current || Boolean(unavailable)
    || !studioToolReady(tools, instruction, recipe, selectionCoverage, activeTool, isolateTool, workflowUnavailable);
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
              {compare.layer && <StudioCompare {...compare.controls} />}
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
              <StudioTryAnother key={current.artifactId} session={session} current={current} busy={busy}
                onTry={(replay) => again(replay, () => setSelectedId(null))} />
              <StudioExportLink artifactId={current.artifactId} />
              {sessionId && !current.isSource && <StudioHideResult onHide={() => { hideStudioStep(sessionId, current.artifactId); setSelectedId(null); }} />}
              {onUseInChat && <StudioUseInChat ready={Boolean(artifact.data)} onUse={() => onUseInChat({ artifactId: current.artifactId, artifact: artifact.data ?? null, origin: current.isSource ? (artifact.data?.original_name ? "uploaded" : "generated") : "edited" })} />}
            </>
          )}
          <StudioCloseButton halfway={cutoutEdit.busy} onClose={onClose} />
        </div>
      </header>
      {/* A save that fails must not look like a save that worked. The button
          only changes on success, so without this the picture silently stays
          unmarked while the label still invites the same press. */}
      {(error || keep.error) && (
        <ErrorCallout message={((error ?? keep.error) as Error).message} />
      )}
      {/* One reason at a time: a cutout that failed says what it left as it was. */}
      {selectionError ? <ErrorCallout message={selectionError} /> : <StudioApplyFailure session={session} />}
      <div className="studio-layout">
        <StudioToolRail
          active={tools.kind}
          selectionKind={tools.selectionKind}
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
              shown={adjustedPreview ?? captionPreview ?? straightenPreview}
              tint={tools.kind === "paint" ? { rgb: paintRgb(tools.paintColor), opacity: tools.paintOpacity / 100 } : null}
              mask={tools.mask}
              tool={pointerTool}
              maskVersion={tools.maskVersion}
              before={compare.layer}
              onGestureStart={() => snapshotBeforeGesture(tools)}
              onStrokeEnd={() => dispatch({ type: "stroke-end" })}
              // Over the picture and at its zoom: the frame is the control,
              // so it has to be where the frame is.
              overlay={tools.kind === "extend"
                ? (shown) => <StudioExtendHandles tools={tools} dispatch={dispatch} picture={bitmap} shown={shown} />
                : undefined}
            />
          ) : (
            <StudioStageLoading error={imageError} reload={reload} />
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
          {toolMarksPicture(tools.kind) && (
            <StudioSelectionControls tools={tools} dispatch={dispatch} coverage={selectionCoverage}
              colorsUnreadable={readsColors && Boolean(bitmap) && !sourcePixels}
              selectSubjectWaits={busy} findingSubject={subjectSearch.busy}
              onSelectSubject={subjectWorkflow && current && bitmap ? () => {
                setSelectionError(null);
                // Kept on the canvas while the cutout joins the strip, so the subject is selected where it was asked for.
                setSelectedId(current.artifactId);
                subjectSearch.start(current.artifactId, { width: bitmap.width, height: bitmap.height }, subjectWorkflow);
              } : undefined} />
          )}
          <StudioToolOptions
            tools={tools}
            dispatch={dispatch}
            instruction={instruction}
            onInstructionChange={(value) => {
              setInstruction(value);
              setRecipe(null);
            }}
            busy={busy}
            onLocalEdit={(operation, details) => current && localEdit(operation, current.artifactId, () => setSelectedId(null), details)}
          />
          <StudioRecipes
            disabled={busy || !current}
            from={studioRecipeSource(session, current)}
            onApply={(chosen) => {
              setRecipe(chosen);
              setInstruction(chosen.instruction);
              // Made on a selection, it needs one: select the way last used, as the rail's Select does.
              if (chosen.mask_mode !== "none" && !toolUsesMask(tools.kind)) dispatch({ type: "select-tool", kind: tools.selectionKind });
            }}
          />
          <StudioRecipeWorkflowNotice recipe={recipe} selected={selectionCoverage > 0} />
          {unavailable && (
            // Beside the button that would fail, and named by the tools that
            // cannot run, so the sentence arrives before the drawing does.
            <StudioToolGuidance reason={unavailable} onOpenWorkflows={onOpenWorkflows} />
          )}
          {unchecked && !EXACT_EDITS.includes(tools.kind) && (
            <StudioCapabilityCheck failed={unchecked === "failed"} onRetry={() => void capabilities.refetch()} />
          )}
          {!EXACT_EDITS.includes(tools.kind) && <StudioResultCount kind={tools.kind} value={results} onChange={setResults} />}
          {!EXACT_EDITS.includes(tools.kind) && (
            <button
              className="primary"
              aria-disabled={applyDisabled}
              onClick={() => {
                if (applyDisabled || !current) return;
                applyStudioEdit({
                  tools, instruction, recipe, activeTool, isolateTool, current, bitmap, results, apply,
                  cutout: cutoutEdit,
                  setError: setSelectionError,
                  onAccepted: () => {
                    setInstruction("");
                    setSelectedId(null);
                  },
                });
              }}
            >
              {studioApplyLabel(tools, busy, selectionCoverage)}
            </button>
          )}
          {applyProgress && <StudioRunningEdit part={applyProgress.part} place={applyProgress.place} stopping={stopping} onStop={stop} />}
        </aside>
      </div>
      <StudioFilmstrip
        steps={steps} generationIdentity={previewArtifactId ? null : current?.generationIdentity ?? artifact.data?.generation_identity}
        selectedId={previewArtifactId ? null : current?.artifactId ?? null}
        onSelect={setSelectedId}
        hidden={hidden} onShowHidden={() => sessionId && showStudioSteps(sessionId)}
      />
    </div>
  );
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

function StudioRecipeWorkflowNotice({ recipe, selected }: { recipe: EditTemplate | null; selected: boolean }) {
  // What the recipe was made on has to be there again before it can run.
  const needs = recipe && recipe.mask_mode !== "none" && !selected
    ? recipe.mask_mode === "inverse"
      ? `${recipe.name} changes everything outside a selection. Select what to keep first.`
      : `${recipe.name} changes only a selected part. Select it first.`
    : null;
  if (!recipe?.workflow_revision_id && !needs) return null;
  return (
    <>
      {recipe?.workflow_revision_id && <small role="status">{recipe.name} supplies the workflow for this edit.</small>}
      {needs && <small role="status">{needs}</small>}
    </>
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
  hidden,
  onShowHidden,
}: {
  steps: StudioStep[];
  selectedId: string | null;
  generationIdentity?: GenerationIdentity | null;
  onSelect: (artifactId: string) => void;
  /** Results taken out of the strip. They keep their numbers, so a branch still names the step it came from. */
  hidden: ReadonlySet<string>;
  onShowHidden: () => void;
}) {
  if (steps.length === 0) return null;
  const hiddenCount = steps.filter((step) => !step.isSource && hidden.has(step.artifactId)).length;
  // The results the chosen one was made from, so its path back is visible among the branches.
  const madeFrom = studioStepAncestors(steps, steps.findIndex((step) => step.artifactId === selectedId));
  // A group of buttons, not a listbox: a real listbox owns focus with a
  // roving tabindex and aria-activedescendant, and role="option" would
  // override the native button role so these stop announcing as activatable.
  return (
    <>
      <div className="studio-filmstrip" role="group" aria-label="Edit history">
      {steps.map((step, index) => {
        if (!step.isSource && hidden.has(step.artifactId)) return null;
        const origin = studioStepOrigin(steps, index);
        return (
          <button
            key={`${step.messageId}-${step.artifactId}`}
            aria-pressed={step.artifactId === selectedId}
            className={step.artifactId === selectedId ? "selected" : madeFrom.has(index) ? "made-from" : ""}
            onClick={() => onSelect(step.artifactId)}
            onFocus={(event) => event.currentTarget.scrollIntoView({ block: "nearest", inline: "nearest" })}
          >
            <img
              src={`/api/artifacts/${encodeURIComponent(step.artifactId)}/content`}
              alt={step.isSource ? "The original image" : `Result of step ${index}`}
              loading="lazy"
            />
            <small>{step.isSource ? "Original" : step.instruction || `Step ${index}`}</small>
            {origin && <small className="studio-step-origin">{origin}</small>}
            {madeFrom.has(index) && <small className="sr-only">Part of how the chosen result was made</small>}
          </button>
        );
      })}
      </div>
      {hiddenCount > 0 && (
        <button type="button" className="secondary compact-button" onClick={onShowHidden}>
          Show {hiddenCount} hidden {hiddenCount === 1 ? "result" : "results"}
        </button>
      )}
      <GenerationIdentitySummary identity={generationIdentity} />
    </>
  );
}
