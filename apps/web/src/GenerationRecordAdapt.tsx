import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { discardBlankChat } from "./discardBlankChat";
import {
  adaptationFailureText,
  adaptationNeeds,
  type AdaptationNeeds,
  type GenerationRecordRequirement,
  type ReplayPlan,
} from "./generationRecord";
import { LibraryImagePicker } from "./LibraryImagePicker";
import { GenerationRecordAdaptChoices } from "./GenerationRecordAdaptChoices";
import { ProfilePicker } from "./ProfilePicker";
import { useProfileLibrary } from "./useProfileLibrary";
import type { ArtifactLibraryItem } from "./types";

/** A new version of a checked record, when only its workflow, model, LoRAs or pictures keep it from being made again here.
 *
 * It is a new generation, never this one made again: the person chooses what
 * stands in for each requirement that does not match, and the run keeps which
 * record it came from and what differs. The record's words, seed and settings
 * go as they are, and so does every picture that is here.
 */
export function GenerationRecordAdapt({
  plan,
  requirements,
  bundledInputs,
  content,
  onStarted,
}: {
  plan: ReplayPlan;
  /** What the record names, in its own order, and whether each is here. */
  requirements: GenerationRecordRequirement[];
  /** The positions of the inputs the checked file carries a copy of. */
  bundledInputs: number[];
  content: ArrayBuffer;
  onStarted: (chatId: string) => void;
}) {
  const needs = adaptationNeeds(plan);
  if (!needs) return null;
  return <AdaptChooser plan={plan} needs={needs} requirements={requirements} bundledInputs={bundledInputs}
    content={content} onStarted={onStarted} />;
}

/** What stands in for a missing input: a picture from the library, or the copy the file carries. */
type InputChoice = { kind: "library"; item: ArtifactLibraryItem } | { kind: "bundle" };

function AdaptChooser({
  plan,
  needs,
  requirements,
  bundledInputs,
  content,
  onStarted,
}: {
  plan: ReplayPlan;
  needs: AdaptationNeeds;
  requirements: GenerationRecordRequirement[];
  bundledInputs: number[];
  content: ArrayBuffer;
  onStarted: (chatId: string) => void;
}) {
  const client = useQueryClient();
  const role = plan.operation === "text_to_video" || plan.operation === "image_to_video" ? "video" : "image";
  const loraCount = requirements.filter((item) => item.kind === "lora").length;
  // Positions count the record's inputs in its own order, as the server reads them.
  const missingInputs = requirements
    .filter((item) => item.kind === "input")
    .flatMap((item, position) => (item.state === "present" ? [] : [{ position, role: item.role }]));
  const [revisionId, setRevisionId] = useState("");
  const [profileId, setProfileId] = useState("");
  const [workflowReady, setWorkflowReady] = useState(true);
  const profiles = useProfileLibrary(role, profileId, needs.model);
  const [leaveOutLoras, setLeaveOutLoras] = useState(false);
  const [pictures, setPictures] = useState<Record<number, InputChoice>>({});
  const [choosingFor, setChoosingFor] = useState<number | null>(null);
  const adapt = useMutation({
    mutationFn: async () => {
      const chat = await api.createChat(null);
      try {
        await api.adaptGenerationRecord(chat.id, content, {
          workflowRevisionId: needs.workflow ? revisionId : undefined,
          profileId: needs.model ? profileId : undefined,
          loras: needs.loras && leaveOutLoras ? Array.from({ length: loraCount }, (_, position) => `${position}:omit`) : [],
          inputs: needs.inputs
            ? missingInputs.flatMap(({ position }) => {
              const choice = pictures[position];
              if (!choice) return [];
              return [choice.kind === "bundle" ? `${position}:bundle` : `${position}:${choice.item.id}`];
            })
            : [],
        });
      } catch (error) {
        // The new chat holds nothing yet, so it goes rather than stays behind empty.
        await discardBlankChat(chat.id);
        throw error;
      }
      return chat.id;
    },
    onSuccess: (chatId) => {
      void client.invalidateQueries({ queryKey: ["chats"] });
      onStarted(chatId);
    },
  });
  const chosen = (!needs.workflow || Boolean(revisionId)) && (!needs.model || Boolean(profileId))
    && (!needs.loras || leaveOutLoras)
    && (!needs.inputs || missingInputs.every(({ position }) => Boolean(pictures[position])));
  const usable = !adapt.isPending && chosen && (!needs.workflow || workflowReady)
    && (!needs.model || profiles.identity.ready);

  return (
    <section className="generation-record-body" aria-labelledby={`adapt-${plan.digest}`}>
      <h3 id={`adapt-${plan.digest}`}>Make a new version</h3>
      <p>
        A new version can be made here with what you choose in place of what does not match. It is a new
        generation, not this one made again, and it keeps which record it came from and what differs.
      </p>
      {needs.workflow && (
        <GenerationRecordAdaptChoices operation={plan.operation} value={revisionId} saving={adapt.isPending}
          onChange={setRevisionId} onReady={setWorkflowReady} />
      )}
      {needs.model && (
        <ProfilePicker library={profiles} label="Model" searchLabel="Search models" value={profileId}
          selectedId={profileId} onChange={setProfileId} disabled={adapt.isPending}>
          <option value="">Choose a model</option>
        </ProfilePicker>
      )}
      {needs.loras && (
        <label>
          <input type="checkbox" checked={leaveOutLoras}
            onChange={(event) => setLeaveOutLoras(event.target.checked)} />
          <span>Leave out its LoRAs</span>
        </label>
      )}
      {needs.inputs && missingInputs.map(({ position, role: inputRole }) => (
        <div key={position}>
          <span>{inputText(position, inputRole)}: {choiceText(pictures[position])}</span>
          {/* Pressed rather than removed once chosen, so keyboard focus stays on it. */}
          {bundledInputs.includes(position) && (
            <button type="button" className="secondary compact-button"
              aria-pressed={pictures[position]?.kind === "bundle"}
              onClick={() => setPictures((current) => ({ ...current, [position]: { kind: "bundle" } }))}>
              Use the copy saved with the record
            </button>
          )}
          <button type="button" className="secondary compact-button"
            onClick={() => setChoosingFor(position)}>
            {pictures[position] ? "Choose another picture" : "Choose a picture"}
          </button>
        </div>
      ))}
      {choosingFor !== null && (
        <LibraryImagePicker
          title={`A picture in place of ${inputText(choosingFor, missingInputs.find((item) => item.position === choosingFor)?.role ?? null).toLowerCase()}`}
          confirmLabel="Use this picture"
          single
          onConfirm={(items) => {
            const [item] = items;
            if (item) setPictures((current) => ({ ...current, [choosingFor]: { kind: "library", item } }));
            setChoosingFor(null);
          }}
          onClose={() => setChoosingFor(null)}
        />
      )}
      {adapt.isPending && <p role="status">Starting it in a new chat…</p>}
      {adapt.isError && <p role="alert">{adaptationFailureText(adapt.error)}</p>}
      <div>
        <button type="button" className="secondary" aria-disabled={!usable} onClick={() => {
          if (usable) adapt.mutate();
        }}>
          Make a new version
        </button>
      </div>
    </section>
  );
}

/** What has been chosen for one missing input, said plainly. */
function choiceText(choice: InputChoice | undefined): string {
  if (!choice) return "not here";
  if (choice.kind === "bundle") return "the copy saved with the record";
  return choice.item.original_name ?? "a picture from the library";
}

/** How a record's input is named to the person: the picture it changed, one it read beside it, or one it was given. */
function inputText(position: number, role: string | null): string {
  if (role === "source" || role === "edit_source") return "The picture it started from";
  if (role === "reference") return `Reference picture ${position + 1}`;
  if (role === "mask") return "Its selection";
  return `Input picture ${position + 1}`;
}
