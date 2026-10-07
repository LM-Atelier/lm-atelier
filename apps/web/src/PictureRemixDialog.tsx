import { Fragment, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { discardBlankChat } from "./discardBlankChat";
import { ReadyWorkflowBrowseControls } from "./ReadyWorkflowBrowseControls";
import { settingLabel } from "./pictureSettings";
import {
  applicableClaims,
  claimKeys,
  readRemixPreview,
  remixClaimText,
  remixFailureText,
  type RemixPreview,
  type RemixRole,
} from "./pictureRemix";
import { useReadyWorkflowChoices } from "./useReadyWorkflowChoices";
import { useProfileLibrary } from "./useProfileLibrary";
import { ProfilePicker } from "./ProfilePicker";

/** The settings a remix would run with, named the way the person reads them. */
const SHOWN_SETTINGS: [string, string][] = [
  ["negative_prompt", "Negative prompt"],
  ["seed", "Seed"],
  ["steps", "Steps"],
  ["cfg", "Guidance"],
  ["sampler", "Sampler"],
  ["scheduler", "Scheduler"],
  ["width", "Width"],
  ["height", "Height"],
];

/** What one choice is called where it is offered. */
function choiceLabel(choice: string, shape: RemixPreview["shape"]): string {
  if (choice === "size") return "Size";
  if (choice === "shape" && shape) return `This picture's shape, at ${shape.width} × ${shape.height}`;
  return settingLabel(choice);
}

/** Where the strength of a change to the picture came from, in words. */
function strengthSource(strength: NonNullable<NonNullable<RemixPreview["resolved"]>["strength"]>): string {
  if (strength.from_file) return "the strength this picture was made with";
  return strength.mode === "auto" ? "estimated from the words" : "the model's own";
}

function shownValue(value: unknown): string {
  if (value === "") return "none";
  return typeof value === "string" || typeof value === "number" ? String(value) : "set";
}

function ResolvedSettings({ preview }: { preview: RemixPreview }) {
  const resolved = preview.resolved;
  if (!resolved) return null;
  const edit = preview.role === "edit";
  // A change to the picture keeps its size, so no size of the workflow's is shown.
  const shown = edit
    ? SHOWN_SETTINGS.filter(([key]) => key !== "width" && key !== "height")
    : SHOWN_SETTINGS;
  return (
    <>
      <strong>{edit ? "A remix would change this picture with" : "A remix would make one picture with"}</strong>
      <dl className="generation-identity">
        {edit && preview.source && <>
          <dt>Starts from</dt>
          <dd>{`this picture, ${preview.source.width} × ${preview.source.height}`}</dd>
        </>}
        <dt>Prompt</dt><dd>{resolved.text}</dd>
        {edit && resolved.strength && <>
          <dt>Strength of the change</dt>
          <dd>{`${resolved.strength.value}, ${strengthSource(resolved.strength)}`}</dd>
        </>}
        {shown.map(([key, label]) => (
          <Fragment key={key}>
            <dt>{label}</dt>
            <dd>{key in resolved.settings
              ? shownValue(resolved.settings[key])
              : key === "seed" && resolved.seed_drawn ? "chosen when it is made" : "the workflow's own"}</dd>
          </Fragment>
        ))}
        {resolved.trigger_words.length > 0 && <>
          <dt>Words the model adds</dt><dd>{resolved.trigger_words.join(", ")}</dd>
        </>}
        {edit && <>
          <dt>Words the engine is given</dt><dd>{resolved.engine_prompt}</dd>
        </>}
      </dl>
    </>
  );
}

/** Check a picture's own settings against a workflow and model chosen here, then make one picture. */
export function PictureRemixDialog({
  artifactId,
  onClose,
  onOpenChat,
}: {
  artifactId: string;
  onClose: () => void;
  onOpenChat?: (chatId: string) => void;
}) {
  const client = useQueryClient();
  const [revisionId, setRevisionId] = useState("");
  const [profileId, setProfileId] = useState("");
  const [applied, setApplied] = useState<string[]>([]);
  const [role, setRole] = useState<RemixRole>("words");
  const workflows = useReadyWorkflowChoices(
    revisionId ? [revisionId] : [], role === "edit" ? "image_to_image" : "text_to_image",
  );
  const profiles = useProfileLibrary("image", profileId);
  const apply = applied.flatMap(claimKeys);
  const preview = useQuery({
    queryKey: ["picture-remix-preview", artifactId, revisionId, profileId, apply, role],
    queryFn: async ({ signal }) => readRemixPreview(await api.remixPreview(
      artifactId, { workflow_revision_id: revisionId, profile_id: profileId, apply, role }, signal,
    )),
    enabled: Boolean(revisionId && profileId),
    retry: false,
    // It holds the picture's prompt, and it is only true for the choices just made.
    staleTime: 0,
    gcTime: 0,
    // Kept on screen while a changed choice of settings is checked, so nothing the
    // person is using disappears under them; never across a workflow or model.
    placeholderData: (previous, previousQuery) => previousQuery
      && previousQuery.queryKey[2] === revisionId && previousQuery.queryKey[3] === profileId
      && previousQuery.queryKey[5] === role
      ? previous : undefined,
  });
  const make = useMutation({
    mutationFn: async (digest: string) => {
      const chat = await api.createChat(null);
      try {
        await api.remixPicture(chat.id, {
          artifact_id: artifactId,
          workflow_revision_id: revisionId,
          profile_id: profileId,
          apply,
          role,
          review_digest: digest,
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
      onOpenChat?.(chatId);
      onClose();
    },
    // What would run may have changed, so what is shown is checked again.
    onError: () => { void preview.refetch(); },
  });
  const choose = (setter: (value: string) => void) => (value: string) => {
    // A remix being started is the one chosen; nothing changes under it.
    if (make.isPending) return;
    // What can be applied depends on the workflow and model, so a new choice starts empty.
    setApplied([]);
    make.reset();
    setter(value);
  };
  const chooseRole = (value: RemixRole) => {
    if (make.isPending || value === role) return;
    // The workflows differ: one makes a picture from words, the other changes one.
    choose(setRevisionId)("");
    setRole(value);
  };
  const data = preview.data;
  const checking = preview.isFetching;
  const applicable = data ? applicableClaims(data.claims, data.shape) : [];
  // Only an answer just checked for exactly these choices can be made; a failed
  // check leaves the earlier answer on screen, which is no longer current.
  const digest = data?.ready && !preview.isPlaceholderData && !preview.isError
    ? data.review_digest : null;
  const usable = Boolean(digest) && !checking && !make.isPending;

  return (
    <AccessibleDialog
      title="Remix these settings"
      eyebrow="Settings in the file"
      closeLabel="Close remix"
      onClose={onClose}
      className="generation-record-dialog"
    >
      <section className="generation-record-body">
        <p>
          Choose a workflow and a model here. Nothing the file names is looked up or fetched;
          each of its settings is checked against what you choose.
        </p>
        <fieldset>
          <legend>Start from</legend>
          {([["words", "Words only"], ["edit", "This picture"]] as const).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="remix-start"
                checked={role === value}
                aria-disabled={make.isPending}
                onChange={() => chooseRole(value)}
              />
              <span>{label}</span>
            </label>
          ))}
        </fieldset>
        <ReadyWorkflowBrowseControls workflows={workflows} />
        <label>
          <span>Workflow</span>
          <select value={revisionId} aria-disabled={make.isPending}
            onChange={(event) => choose(setRevisionId)(event.target.value)}>
            <option value="">Choose a workflow</option>
            {revisionId && !workflows.rows.some((row) => row.revision_id === revisionId) && (
              <option value={revisionId}>{workflows.missingLabel}</option>
            )}
            {workflows.rows.map((row) => (
              <option key={row.revision_id} value={row.revision_id}>
                {`${row.family_name} - ${row.workflow_name} · v${row.revision_version}`}
              </option>
            ))}
          </select>
        </label>
        <ProfilePicker library={profiles} label="Model" searchLabel="Search models" value={profileId}
          selectedId={profileId} onChange={choose(setProfileId)} disabled={make.isPending}>
          <option value="">Choose a model</option>
        </ProfilePicker>
        {checking && <p role="status">Checking these choices…</p>}
        {preview.isError && <p role="alert">These choices could not be checked.</p>}
        {data && !data.ready && data.refusals.map((refusal) => (
          <p key={refusal.code} role="alert">{refusal.message}</p>
        ))}
        {data && data.claims.length > 0 && (
          <ul>
            {data.claims.map((item) => (
              <li key={item.key}>
                <strong>{settingLabel(item.key)}</strong>: {String(item.value)}. {item.key === "prompt"
                  ? "Used as the words the picture is made from."
                  : `It ${remixClaimText(item)}.`}
              </li>
            ))}
          </ul>
        )}
        {applicable.length > 0 && (
          <fieldset aria-busy={checking}>
            <legend>Use from the file</legend>
            {applicable.map((choice) => (
              <label key={choice}>
                <input
                  type="checkbox"
                  checked={applied.includes(choice)}
                  // Never disabled, which would drop focus out of the dialog.
                  aria-disabled={checking || make.isPending}
                  onChange={(event) => {
                    if (checking || make.isPending) return;
                    const on = event.target.checked;
                    make.reset();
                    setApplied((current) => on
                      ? [...current, choice]
                      : current.filter((key) => key !== choice));
                  }}
                />
                <span>{choiceLabel(choice, data?.shape ?? null)}</span>
              </label>
            ))}
          </fieldset>
        )}
        {data && <ResolvedSettings preview={data} />}
        {make.isPending && <p role="status">Starting it in a new chat…</p>}
        {make.isError && <p role="alert">{remixFailureText(make.error)}</p>}
        <p>
          A remix is one new picture in a new chat. The picture and its file are not changed, and
          only which picture and choices it came from are kept with it.
        </p>
        <div>
          <button type="button" className="secondary" aria-disabled={!usable} onClick={() => {
            if (usable && digest) make.mutate(digest);
          }}>
            Make this picture
          </button>
        </div>
      </section>
    </AccessibleDialog>
  );
}
