import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import { ApiError, api } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { formatBytes } from "./format";
import type { ArtifactCleanupResult, RetentionPolicy } from "./types";

const DAY_CHOICES = [7, 14, 30, 90, 180, 365];
const HOUR_CHOICES = [6, 12, 24, 72, 168];

type Windows = { days: number; hours: number };
/** A choice, and the revision of the choice it was made from. */
type Proposal = Windows & { revision: number };

function count(value: number, one: string, many: string): string {
  return `${value.toLocaleString()} ${value === 1 ? one : many}`;
}

function daysLabel(days: number): string {
  return days % 365 === 0 ? count(days / 365, "year", "years") : count(days, "day", "days");
}

function hoursLabel(hours: number): string {
  return hours % 24 === 0 ? count(hours / 24, "day", "days") : count(hours, "hour", "hours");
}

/** The usual choices, plus whatever is in force or the installation's default, in order. */
function choices(presets: number[], ...kept: number[]): number[] {
  return [...new Set([...presets, ...kept])].sort((a, b) => a - b);
}

/** How long the workspace keeps files nothing uses, chosen for the whole workspace.
 *
 * The server keeps the choice, so every open window and every clearing pass
 * uses the same one. Choosing clears nothing by itself: files that become ready
 * are cleared the next time the workspace starts, or with Clear now.
 *
 * A shorter window can make files ready that were not, so it is checked with the
 * server first, and if the server finds anything it would clear, the choice
 * waits for a confirmation that says how much. The figures already on screen
 * are not asked: something may have been cleared since they were read, and a
 * smaller count then would let newly ready files through unannounced. A choice
 * another window changed first is refused, and the choice in force is shown
 * instead.
 */
export function RetentionChoices() {
  const id = useId();
  const client = useQueryClient();
  const policy = useQuery({ queryKey: ["retention-policy"], queryFn: api.retentionPolicy });
  const [proposal, setProposal] = useState<(Proposal & { found: ArtifactCleanupResult | null }) | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const settle = () => {
    setProposal(null);
    void client.invalidateQueries({ queryKey: ["artifact-storage"] });
  };

  const choose = useMutation({
    mutationFn: ({ revision, days, hours }: Proposal) => api.chooseRetention(revision, days, hours),
    onSuccess: (chosen: RetentionPolicy) => client.setQueryData(["retention-policy"], chosen),
    onError: (error) => {
      if (error instanceof ApiError && error.code === "retention-policy-stale") {
        setProblem("Retention was changed in another window, so nothing was saved. This is the choice in force now.");
        void client.invalidateQueries({ queryKey: ["retention-policy"] });
        return;
      }
      setProblem(error.message);
    },
    onSettled: settle,
  });

  const check = useMutation({
    mutationFn: ({ days, hours }: Proposal) => api.previewRetention(days, hours),
    onSuccess: (found, proposed) => {
      if (found.removed_count > 0) {
        setProposal({ ...proposed, found });
        return;
      }
      // Still the revision the choice was made from, even if the choice in
      // force has been read again since: a newer one must refuse this save.
      choose.mutate(proposed);
    },
    onError: (error) => {
      setProblem(error.message);
      setProposal(null);
    },
  });

  if (policy.isError) {
    return <p className="muted">Retention choices are unavailable right now.</p>;
  }
  if (!policy.data) return null;
  const current = policy.data;
  const busy = proposal !== null || check.isPending || choose.isPending;
  const shown: Windows = proposal ?? { days: current.media_days, hours: current.temporary_hours };

  const propose = (next: Windows) => {
    if (busy) return;
    if (next.days === current.media_days && next.hours === current.temporary_hours) return;
    const proposed: Proposal = { ...next, revision: current.revision };
    setProblem(null);
    setProposal({ ...proposed, found: null });
    if (next.days < current.media_days || next.hours < current.temporary_hours) {
      check.mutate(proposed);
      return;
    }
    // Keeping files longer never makes one ready that was not.
    choose.mutate(proposed);
  };

  const labelled = (value: number, label: string, fallback: number) =>
    value === fallback ? `${label} (usual)` : label;

  return (
    <section aria-labelledby={`${id}-heading`}>
      <div className="detail-title">
        <div>
          <h2 id={`${id}-heading`}>Retention</h2>
          <p>How long files nothing uses are kept before they are cleared. Favorites are always kept.</p>
        </div>
      </div>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-media`}>Images and videos</strong>
          <small id={`${id}-media-help`}>Kept this long after nothing uses them.</small>
        </span>
        <select
          aria-labelledby={`${id}-media`}
          aria-describedby={`${id}-media-help`}
          aria-disabled={busy}
          value={shown.days}
          onChange={(event) => propose({ days: Number(event.target.value), hours: shown.hours })}
        >
          {choices(DAY_CHOICES, current.media_days, current.default_media_days).map((days) => (
            <option key={days} value={days}>
              {labelled(days, daysLabel(days), current.default_media_days)}
            </option>
          ))}
        </select>
      </div>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-previews`}>Previews and in-between steps</strong>
          <small id={`${id}-previews-help`}>Kept this long after they are made, unless something uses them.</small>
        </span>
        <select
          aria-labelledby={`${id}-previews`}
          aria-describedby={`${id}-previews-help`}
          aria-disabled={busy}
          value={shown.hours}
          onChange={(event) => propose({ days: shown.days, hours: Number(event.target.value) })}
        >
          {choices(HOUR_CHOICES, current.temporary_hours, current.default_temporary_hours).map((hours) => (
            <option key={hours} value={hours}>
              {labelled(hours, hoursLabel(hours), current.default_temporary_hours)}
            </option>
          ))}
        </select>
      </div>
      {problem && (
        <div className="callout error" role="alert">
          {problem}
        </div>
      )}
      {proposal?.found && (
        <ConfirmDialog
          title="Clear files sooner?"
          question={`${count(proposal.found.removed_count, "file", "files")}, ${formatBytes(proposal.found.reclaimed_bytes)}, would be ready to clear.`}
          detail={
            <p>
              {`Images and videos would be kept ${daysLabel(proposal.days)} after nothing uses them, `}
              {`and previews ${hoursLabel(proposal.hours)}. `}
              Ready files are cleared the next time LM Atelier starts, or with Clear now, and cannot be restored.
            </p>
          }
          confirmLabel={choose.isPending ? "Saving…" : "Keep them for less"}
          confirmDisabled={choose.isPending}
          onConfirm={() => choose.mutate({ days: proposal.days, hours: proposal.hours, revision: proposal.revision })}
          onCancel={() => {
            if (!choose.isPending) setProposal(null);
          }}
        />
      )}
    </section>
  );
}
