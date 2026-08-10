import type { GenerationIdentity } from "./types";

const MAX_CAPTURED_NAME = 500;

function capturedName(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const name = value.trim();
  return name && name.length <= MAX_CAPTURED_NAME ? name : null;
}

/** A display-only projection of the immutable witness captured for one run.
 * Raw provenance contains paths, manifests, and resolver details that do not
 * belong in a reusable UI contract. */
export function generationIdentityFromProvenance(
  value: unknown,
): GenerationIdentity | null {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    return null;
  const provenance = value as Record<string, unknown>;
  const model =
    typeof provenance.model === "object" &&
    provenance.model !== null &&
    !Array.isArray(provenance.model)
      ? (provenance.model as Record<string, unknown>)
      : null;
  const workflow =
    typeof provenance.workflow === "object" &&
    provenance.workflow !== null &&
    !Array.isArray(provenance.workflow)
      ? (provenance.workflow as Record<string, unknown>)
      : null;
  const familyName = capturedName(workflow?.family_name);
  const definitionName = capturedName(workflow?.definition_name);
  const rawVersion = workflow?.version;
  const workflowVersion =
    typeof rawVersion === "number" &&
    Number.isSafeInteger(rawVersion) &&
    rawVersion > 0 &&
    rawVersion <= 2_147_483_647 &&
    Boolean(familyName || definitionName)
      ? rawVersion
      : null;
  const identity: GenerationIdentity = {
    model_profile_name: capturedName(model?.profile_name),
    workflow_family_name: familyName,
    workflow_definition_name: definitionName,
    workflow_version: workflowVersion,
  };
  return identity.model_profile_name
    || identity.workflow_family_name
    || identity.workflow_definition_name
    ? identity
    : null;
}
