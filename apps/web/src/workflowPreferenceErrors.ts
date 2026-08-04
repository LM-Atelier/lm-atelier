/** What the server refuses, said the way a person would say it. */
export function preferenceRefusal(code: string | undefined): string | null {
  if (code === "workflow-default-disabled") {
    return "A workflow has to be offered for this before it can be the default one.";
  }
  if (code === "workflow-family-not-selectable") {
    return "This workflow is archived or turned off. Restore it before offering it here.";
  }
  if (code === "workflow-family-operation-unavailable") {
    return "This workflow cannot handle that kind of request.";
  }
  if (code === "workflow-preference-in-use") {
    return "Another saved choice uses this workflow here. Change that choice before turning it off.";
  }
  return null;
}
