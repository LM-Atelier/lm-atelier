import type { GenerationIdentity } from "./types";

function workflowLabel(identity: GenerationIdentity): string | null {
  const names = [
    identity.workflow_family_name,
    identity.workflow_definition_name,
  ].filter(
    (name, index, all): name is string =>
      Boolean(name) && all.indexOf(name) === index,
  );
  if (identity.workflow_version !== null)
    names.push(`v${identity.workflow_version}`);
  return names.length ? names.join(" · ") : null;
}

export function GenerationIdentitySummary({
  identity,
}: {
  identity?: GenerationIdentity | null;
}) {
  if (!identity) return null;
  const workflow = workflowLabel(identity);
  if (!identity.model_profile_name && !workflow) return null;
  return (
    <dl className="generation-identity" aria-label="Generation details">
      {identity.model_profile_name && (
        <>
          <dt>Model</dt>
          <dd>{identity.model_profile_name}</dd>
        </>
      )}
      {workflow && (
        <>
          <dt>Workflow</dt>
          <dd>{workflow}</dd>
        </>
      )}
    </dl>
  );
}
