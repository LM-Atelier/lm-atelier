import type { Ref } from "react";
import { formatBytes } from "./format";
import { refusalSubject, SEED_POLICY_LABELS, settingRows } from "./generationComparison";
import type {
  ArmPreflight,
  ExperimentArm,
  ExperimentRefusal,
  ResourceEvidence,
  SeedPolicyKind,
} from "./generationExperimentTypes";
import type { CheckedComparison } from "./useGenerationComparison";
import "./GenerationComparisonView.css";

/** Why a comparison cannot run, one line per reason, with the one alternative a reason offers. */
export function ComparisonRefusals({ refusals, labels, onSeedPolicy, onProfile }: {
  refusals: ExperimentRefusal[];
  labels: string[];
  onSeedPolicy?: (kind: SeedPolicyKind) => void;
  onProfile?: (armOrdinal: number, profileId: string) => void;
}) {
  if (!refusals.length) return null;
  return <ul className="comparison-refusals">
    {refusals.map((refusal, index) => <li key={`${refusal.code}-${refusal.arm_ordinal ?? "all"}-${index}`}>
      <strong>{refusalSubject(refusal, labels)}</strong>{": "}{refusal.message}
      {refusal.setting && <span>{` Setting: ${refusal.setting}.`}</span>}
      {refusal.alternative?.seed_policy && onSeedPolicy && <button type="button" className="secondary compact-button"
        onClick={() => onSeedPolicy(refusal.alternative?.seed_policy as SeedPolicyKind)}>
        {`Use “${SEED_POLICY_LABELS[refusal.alternative.seed_policy]}” instead`}</button>}
      {refusal.alternative?.profile_id && refusal.arm_ordinal && onProfile && <button type="button" className="secondary compact-button"
        onClick={() => onProfile(refusal.arm_ordinal as number, refusal.alternative?.profile_id as string)}>
        Use the model this workflow expects</button>}
    </li>)}
  </ul>;
}

/** One choice as it resolved: its model, workflow version, size, family and trigger words. */
export function ComparisonChoiceSummary({ arm, seed }: { arm: ArmPreflight | ExperimentArm; seed?: number | null }) {
  return <dl className="comparison-summary">
    <dt>Model</dt><dd>{arm.profile_name ?? arm.profile_id}</dd>
    <dt>Workflow</dt><dd>{arm.workflow_version ? `Version ${arm.workflow_version}` : arm.workflow_revision_id}</dd>
    <dt>Size</dt><dd>{arm.width && arm.height ? `${arm.width} × ${arm.height}` : "Not worked out"}</dd>
    <dt>Model family</dt><dd>{arm.model_family ?? "Not known"}</dd>
    {seed !== undefined && seed !== null && <><dt>Seed</dt><dd>{seed}</dd></>}
    <dt>Trigger words</dt><dd>{arm.trigger_words_applied.length
      ? `Added to this choice's prompt: ${arm.trigger_words_applied.join(", ")}` : "No trigger words added"}</dd>
  </dl>;
}

/** Each setting both choices resolved to, side by side, with differences marked in words. */
export function ComparisonSettings({ arms }: { arms: (ArmPreflight | ExperimentArm)[] }) {
  if (arms.length !== 2) return null;
  const rows = settingRows(arms[0], arms[1]);
  return <table className="comparison-settings">
    <caption>Settings each choice runs with</caption>
    <thead><tr><th scope="col">Setting</th><th scope="col">{arms[0].label}</th><th scope="col">{arms[1].label}</th><th scope="col">Same?</th></tr></thead>
    <tbody>{rows.map((row) => <tr key={row.key}>
      <th scope="row">{row.key}</th><td>{row.values[0]}</td><td>{row.values[1]}</td>
      <td className={row.differs ? "comparison-differs" : undefined}>{row.differs ? "Differs" : "Same"}</td>
    </tr>)}</tbody>
  </table>;
}

export function ComparisonEstimate({ estimate, nouns = "pictures" }: { estimate: ResourceEvidence[]; nouns?: "pictures" | "videos" }) {
  if (!estimate.length) return null;
  return <ul className="comparison-estimate">{estimate.map((item) => <li key={item.resource}>
    {item.resource === "output_bytes" ? `About ${formatBytes(item.value)} of ${nouns}` : `${item.value.toLocaleString()} work units`}
    {" (estimated)"}
  </li>)}</ul>;
}

/** The answer to a check: why it cannot run, or both choices as they would run and Accept. */
export function ComparisonCheckResult({ checked, stale, accepting, onAccept, headingRef, onSeedPolicy, onProfile }: {
  checked: CheckedComparison;
  stale: boolean;
  accepting: boolean;
  onAccept: () => void;
  headingRef: Ref<HTMLHeadingElement>;
  onSeedPolicy: (kind: SeedPolicyKind) => void;
  onProfile: (armOrdinal: number, profileId: string) => void;
}) {
  const { preflight, request } = checked;
  const labels = request.arms.map((arm) => arm.label);
  const blocked = stale || accepting;
  const video = request.operation === "text_to_video";
  return <section className="comparison-check" aria-labelledby="comparison-check-heading">
    <h2 id="comparison-check-heading" ref={headingRef} tabIndex={-1}>
      {preflight.outcome === "compatible" ? "Both choices can run" : "This comparison cannot run as asked"}</h2>
    {stale && <p role="status">Changed since it was checked. Check again before accepting.</p>}
    <ComparisonRefusals refusals={preflight.refusals} labels={labels} onSeedPolicy={onSeedPolicy} onProfile={onProfile} />
    {preflight.outcome === "compatible" && <>
      <p>{preflight.seed_equivalence === "same_family"
        ? "Both choices run one model family, so one seed gives them the same starting point."
        : "The seeds are recorded, but different models do not share a starting point."}</p>
      <div className="comparison-columns">{preflight.arms.map((arm) => <section key={arm.ordinal} aria-label={arm.label}>
        <h3>{arm.label}</h3><ComparisonChoiceSummary arm={arm} /></section>)}</div>
      <ComparisonSettings arms={preflight.arms} />
      <ComparisonEstimate estimate={preflight.estimate} nouns={video ? "videos" : "pictures"} />
      {preflight.confirmation_required && <p>{video ? "These videos will take a while to make; you will be asked to confirm before they are made."
        : "These pictures are large; you will be asked to confirm before they are made."}</p>}
      <button type="button" className="primary" aria-disabled={blocked}
        onClick={() => { if (!blocked) onAccept(); }}>{accepting ? "Accepting…" : "Accept this comparison"}</button>
    </>}
  </section>;
}
