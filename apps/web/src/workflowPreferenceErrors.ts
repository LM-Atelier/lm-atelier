/** What the server refuses, said the way a person would say it. */
export function preferenceRefusal(code: string | undefined): string | null {
  if (code === "workflow-default-disabled") {
    return "A workflow has to be offered for this before it can be the default one.";
  }
  if (code === "workflow-family-not-selectable") {
    return "This workflow is archived or turned off. Restore it before offering it here.";
  }
  return null;
}
