import { formatBytes } from "./format";
import type { HardwareFitAdvice, HardwareFitBasis, HardwareFitStatus } from "./types";

const headings: Record<HardwareFitStatus, string> = {
  recommended: "Recommended for this hardware",
  likely: "Likely to fit",
  tight: "Memory may be tight",
  unsupported: "Hardware not supported",
  unknown: "Hardware fit unknown",
};

const basisLabels: Record<HardwareFitBasis, string> = {
  unknown: "There is not enough information to predict this model's fit.",
  calculated: "Calculated estimate, not a tested result.",
  declared: "Based on declared hardware requirements.",
  measured: "Based on measurements from this exact setup.",
  tested: "Based on measurements from this exact setup.",
  certified: "Based on measurements from this exact setup.",
};

export function InstallHardwareFit({ fit }: { fit: HardwareFitAdvice }) {
  const evidence = fit.evidence_label === fit.basis && fit.status !== "unsupported"
    ? fit.evidence_label
    : null;
  const basis = evidence === "tested"
    ? "Tested on this exact setup."
    : evidence === "certified"
      ? "Certified for this exact setup."
      : basisLabels[fit.basis];

  return (
    <section aria-label="Hardware fit">
      <h3>{headings[fit.status]}</h3>
      <p>{basis}</p>
      {fit.resources.length > 0 && (
        <dl className="install-facts">
          {fit.resources.map((resource) => (
            <div key={resource.kind}>
              <dt>{resource.kind === "system" ? "System memory" : "Accelerator memory"}</dt>
              <dd>
                {formatBytes(resource.required_bytes)} needed · {resource.capacity_bytes > 0 ? `${formatBytes(resource.capacity_bytes)} total` : "capacity unknown"}
                <br />
                <small>{resource.available_bytes == null ? "Free memory not reported" : `${formatBytes(resource.available_bytes)} free now`}</small>
              </dd>
            </div>
          ))}
        </dl>
      )}
      {fit.reasons.map((reason) => (
        <p key={reason.code} className={reason.severity === "info" ? undefined : "install-warning"}>
          {reason.message}
        </p>
      ))}
      {fit.alternatives.length > 0 && (
        <ul>{fit.alternatives.map((alternative) => <li key={alternative.code}>{alternative.message}</li>)}</ul>
      )}
      {fit.settings.length > 0 && (
        <dl className="install-facts">
          {fit.settings.map((setting) => (
            <div key={setting.key}>
              <dt>{setting.label}</dt>
              <dd>
                <small>Suggested range: {setting.minimum}–{setting.maximum} {setting.unit}</small>
                {setting.preserves_user_override && <><br /><small>Your chosen setting will be kept.</small></>}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
