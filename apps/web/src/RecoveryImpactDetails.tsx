import { formatBytes } from "./format";
import type { RecoveryCounts, RecoveryKind } from "./recoveryTypes";

export function RecoveryImpactDetails({ counts, kind }: { counts: RecoveryCounts; kind: RecoveryKind }) {
  if (kind === "project") return <p className="muted">{counts.chats} chats stay available.<br />No chat history or media is deleted.</p>;
  if (kind === "workflow_family") return <p className="muted">
    {counts.workflow_definitions} workflows · {counts.workflow_revisions} revisions
    <br />{counts.references} retained references · {counts.active_work} active work items
    <br />Restore keeps this workflow disabled and does not grant trust or activate it.
  </p>;
  return <p className="muted">
    {counts.messages} messages · {counts.artifacts} media files · {counts.runs} generations
    <br />Retained media: {formatBytes(counts.retained_bytes)} (estimate)
    <br />Eligible for later reclamation: {formatBytes(counts.reclaimable_bytes)}
  </p>;
}
