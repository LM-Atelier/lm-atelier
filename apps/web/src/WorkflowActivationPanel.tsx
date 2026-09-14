import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import {
  activationChoiceGroups,
  activationChoiceKey,
  freshActivationSelections,
  initialActivationChoices,
  selectedActivationDependencies,
} from "./workflowActivationChoices";
import type { WorkflowActivationPreparation, WorkflowActivationSelection } from "./types";
import "./WorkflowActivationPanel.css";

const CHANGED_WORKFLOW = "The workflow changed. Refresh its details before activating dependencies.";
const CHANGED_CHOICES = "Dependencies changed. Check the choices and try again.";

function activationError(error: unknown): string {
  if (error instanceof Error) {
    if ((error as Error & { code?: string }).code === "workflow-activation-unavailable") {
      return "The workflow or its dependencies changed. Refresh the choices and review the workflow again.";
    }
    return error.message;
  }
  return "Dependencies could not be activated. Try refreshing the choices.";
}

export function WorkflowActivationPanel({
  workflowId, revisionId, contractSha256,
}: { workflowId: string; revisionId: string; contractSha256: string }) {
  const client = useQueryClient();
  const [snapshot, setSnapshot] = useState<WorkflowActivationPreparation | null>(null);
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [optional, setOptional] = useState<Record<string, boolean>>({});
  const [pending, setPending] = useState(false);
  const [activated, setActivated] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => { request.current?.abort(); }, [workflowId, revisionId, contractSha256]);

  const selected = snapshot ? selectedActivationDependencies(snapshot, choices, optional) : null;

  async function run(mode: "inspect" | "activate") {
    if (request.current || (mode === "activate" && snapshot && selected === null)) return;
    const controller = new AbortController();
    request.current = controller;
    setPending(true); setError(null); setActivated(false);
    const previous = snapshot;
    let submitted = false;
    try {
      const fresh = await api.prepareWorkflowActivation(workflowId, revisionId, controller.signal);
      if (controller.signal.aborted) return;
      if (fresh.workflow_revision_id !== revisionId
          || fresh.dependency_contract_sha256 !== contractSha256
          || !/^[a-f0-9]{64}$/.test(fresh.workflow_artifact_sha256)) {
        throw new Error(CHANGED_WORKFLOW);
      }
      const reset = () => {
        setSnapshot(fresh);
        setChoices(initialActivationChoices(fresh));
        setOptional({});
      };
      if (mode === "inspect") {
        reset();
        return;
      }
      let selections: WorkflowActivationSelection[] | null;
      if (previous) {
        selections = selected && freshActivationSelections(selected, fresh);
        if (previous.workflow_artifact_sha256 !== fresh.workflow_artifact_sha256 || selections === null) {
          reset();
          throw new Error(CHANGED_CHOICES);
        }
        setSnapshot(fresh);
      } else {
        reset();
        selections = fresh.selections;
        if (fresh.state !== "prepared" || selections === null) return;
      }
      if (controller.signal.aborted) return;
      // Once submitted, keep observing the response after navigation.
      // Cancelling transport cannot undo the server activation.
      submitted = true;
      const result = await api.activateWorkflowRevision(workflowId, revisionId, {
        workflow_artifact_sha256: fresh.workflow_artifact_sha256,
        dependency_contract_sha256: fresh.dependency_contract_sha256,
        selections,
      });
      if (controller.signal.aborted) return;
      if (result.workflow_revision_id !== revisionId
          || result.dependency_contract_sha256 !== contractSha256
          || result.state !== "ready" || !result.is_active) {
        throw new Error("Activation could not be confirmed. Refresh the workflow details.");
      }
      setActivated(true);

    } catch (failure) {
      if (!controller.signal.aborted) setError(activationError(failure));
    } finally {
      if (submitted) {
        await Promise.allSettled(["workflows", "workflow-families", "workflow-family", "studio-capabilities"]
          .map(key => client.invalidateQueries({ queryKey: [key] })));
      }
      if (request.current === controller) request.current = null;
      if (!controller.signal.aborted) setPending(false);
    }
  }

  return (
    <section className="workflow-input-section workflow-activation" aria-label="Workflow dependencies">
      <h3>Installed dependencies</h3>
      <p>Activate this reviewed workflow with matching resources already installed on this computer.</p>
      {pending && <p role="status">Checking dependencies…</p>}
      {error && <ErrorCallout message={error} />}
      {activated && <p role="status">Dependencies activated.</p>}
      {snapshot && snapshot.slots.map(slot => (
        <fieldset key={slot.name} disabled={pending}>
          <legend>{slot.name.replaceAll("_", " ")}</legend>
          {!slot.required && <label className="workflow-activation-optional">
            <input type="checkbox" checked={optional[slot.name] ?? false} onChange={event => {
              setOptional(current => ({ ...current, [slot.name]: event.target.checked }));
              setActivated(false);
            }} />Use {slot.name.replaceAll("_", " ")}
          </label>}
          {(slot.required || optional[slot.name]) && activationChoiceGroups(slot).map(group => (
            <div key={group.key}>
              <label>{group.label}
                <select value={choices[group.key] ?? ""} onChange={event => {
                  setChoices(current => ({ ...current, [group.key]: event.target.value }));
                  setActivated(false);
                }}>
                  <option value="">Choose an installed resource</option>
                  {group.choices.map(choice => <option key={activationChoiceKey(choice.selection)}
                    value={activationChoiceKey(choice.selection)}>{choice.name}</option>)}
                </select>
              </label>
              {group.choices.length === 0 && <p>
                No matching installed resource. Install the required dependency, then refresh.
              </p>}
            </div>
          ))}
        </fieldset>
      ))}
      <div className="row-actions">
        <button type="button" className="primary compact-button"
          disabled={pending || (snapshot !== null && selected === null)}
          onClick={() => { void run("activate"); }}>Activate dependencies</button>
        <button type="button" className="secondary compact-button" disabled={pending}
          onClick={() => { void run("inspect"); }}>
          {snapshot ? "Refresh dependencies" : "Choose dependencies"}
        </button>
      </div>
    </section>
  );
}
