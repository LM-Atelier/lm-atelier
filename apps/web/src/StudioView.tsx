import { Download, Star, X } from "lucide-react";
import { useCallback, useEffect, useId, useMemo, useReducer, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { GenerationIdentitySummary } from "./GenerationIdentitySummary";
import { StudioOpenImage } from "./StudioOpenImage";
import { ErrorCallout } from "./ErrorCallout";
import { StudioCanvas } from "./StudioCanvas";
import { StudioExtendHandles } from "./StudioExtendHandles";
import { StudioRecipes } from "./StudioRecipes";
import { StudioToolGuidance } from "./StudioToolGuidance";
import { StudioToolRail } from "./StudioToolRail";
import { StudioWorkflowSelector } from "./StudioWorkflowSelector";
import { artifactSource } from "./messageMedia";
import { coverage, encodeMaskPng, isEmpty } from "./studioMasks";
import {
  initialToolState,
  defaultInstruction,
  studioToolReducer,
  toolFor,
  toolUsesMask,
  type StudioToolKind,
} from "./studioToolState";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession, type StudioStep } from "./useStudioSession";
import { useConfirm } from "./useConfirm";
import type { EditTemplate, GenerationIdentity } from "./types";

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
  // The pointer tool is rebuilt whenever the mode or brush changes; each one
  // is a cheap wrapper over the shared raster, never a copy of it.
  const pointerTool = useMemo(
    () => toolFor(tools),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tools.kind, tools.brushRadius, tools.mask],
  );
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
  useEffect(() => {
    if (bitmap) {
      dispatch({ type: "image-changed", width: bitmap.width, height: bitmap.height });
    }
  }, [bitmap]);
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
              onGestureStart={() => dispatch({ type: "gesture-start" })}
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
          {tools.kind === "extend" ? (
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
          ) : tools.kind === "enhance" ? (
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
          ) : (
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
                onChange={(event) => {
                  setInstruction(event.target.value);
                  setRecipe(null);
                }}
              />
            </label>
          )}
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
            // Enhance asks for no words: the whole picture is the subject and
            // the size is the whole instruction.
            disabled={
              (tools.kind === "extend" && !Object.values(tools.margins).some(Boolean)) ||
              (tools.kind !== "enhance" && tools.kind !== "extend" && !instruction.trim()) ||
              busy ||
              !current ||
              Boolean(unavailable) ||
              Boolean(workflowUnavailable && !recipe?.workflow_revision_id)
            }
            onClick={() => {
              if (!current) return;
              const selection = toolUsesMask(tools.kind) && tools.mask && !isEmpty(tools.mask)
                ? tools.mask
                : null;
              // Enhance and Extend ask for no words, and the turn requires
              // some: both were reaching the server and being refused before
              // anything ran. The user's words win whenever there are any.
              const words = instruction.trim() || defaultInstruction(tools);
              const send = (mask: Blob | null) => {
                apply(
                  words,
                  current.artifactId,
                  mask ? { blob: mask, featherPx: tools.featherPx, invert: false } : undefined,
                  tools.kind === "enhance"
                    ? { upscale_factor: tools.upscaleFactor }
                    : tools.kind === "extend"
                      ? { outpaint_margins: tools.margins }
                      : recipe
                        ? recipe.settings_json
                        : undefined,
                  recipe?.workflow_revision_id ?? undefined,
                  () => {
                    setInstruction("");
                    setSelectedId(null);
                  },
                );
              };
              if (selection) void encodeMaskPng(selection).then(send);
              else send(null);
            }}
          >
            {busy
              ? "Applying…"
              : tools.kind === "extend"
                ? "Extend"
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
