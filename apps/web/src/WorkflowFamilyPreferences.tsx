import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { preferenceRefusal } from "./workflowPreferenceErrors";
import type {
  WorkflowFamily,
  WorkflowFamilyPreference,
  WorkflowSelectorCapability,
} from "./types";

const CAPABILITIES: Array<{ key: WorkflowSelectorCapability; label: string }> = [
  { key: "chat", label: "Conversation" },
  { key: "vision", label: "Looking at images" },
  { key: "image", label: "Images" },
  { key: "video", label: "Video" },
];

/** Decide where a workflow family is offered, and where it leads.
 *
 * Two separate questions that look like one. Being offered for a kind of
 * request is about whether it belongs in that list at all; being the
 * default is about what happens when nobody chooses. The server keeps one
 * default per capability, so making this one the default is what moves it
 * rather than a second thing to remember to unset.
 */
export function WorkflowFamilyPreferences({ family }: { family: WorkflowFamily }) {
  const client = useQueryClient();
  const save = useMutation({
    mutationFn: ({
      capability,
      preference,
    }: {
      capability: WorkflowSelectorCapability;
      preference: { enabled: boolean; is_default: boolean; sort_order: number };
    }) => api.setWorkflowFamilyPreference(family.id, capability, preference),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["workflow-families"] });
      void client.invalidateQueries({ queryKey: ["workflow-family", family.id] });
    },
  });

  const known = (capability: WorkflowSelectorCapability): WorkflowFamilyPreference =>
    family.preferences.find((one) => one.selector_capability === capability) ?? {
      selector_capability: capability,
      enabled: false,
      is_default: false,
      sort_order: 0,
    };

  const refusal = preferenceRefusal(
    (save.error as { code?: string } | null)?.code,
  );

  return (
    <section className="family-preferences">
      <h3>Where this workflow is offered</h3>
      {save.error && (
        <ErrorCallout message={refusal ?? (save.error as Error).message} />
      )}
      <ul>
        {CAPABILITIES.map(({ key, label }) => {
          const preference = known(key);
          return (
            <li key={key}>
              <label className="toggle-row">
                <span className="toggle-copy">
                  <strong>{label}</strong>
                  {preference.is_default && <small>Used when nobody chooses</small>}
                </span>
                <input
                  type="checkbox"
                  checked={preference.enabled}
                  disabled={save.isPending}
                  onChange={(event) =>
                    save.mutate({
                      capability: key,
                      preference: {
                        enabled: event.target.checked,
                        // Turning it off cannot leave it the default, which
                        // is the same rule the server enforces - better to
                        // agree with it than to be refused by it.
                        is_default: event.target.checked && preference.is_default,
                        sort_order: preference.sort_order,
                      },
                    })}
                />
              </label>
              {preference.enabled && !preference.is_default && (
                <button
                  className="secondary compact-button"
                  disabled={save.isPending}
                  onClick={() =>
                    save.mutate({
                      capability: key,
                      preference: { enabled: true, is_default: true, sort_order: preference.sort_order },
                    })}
                >
                  Make this the default
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
