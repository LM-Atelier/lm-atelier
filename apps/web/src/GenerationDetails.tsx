import { Fragment, useState } from "react";
import { GenerationIdentitySummary } from "./GenerationIdentitySummary";
import { recordedGenerationDetails, type RecordedLora } from "./generationConfiguration";

function LoraDetails({ lora }: { lora: RecordedLora }) {
  return <li>
    <strong>{lora.name}</strong>
    <small>
      {lora.enabled === false ? "Disabled · " : lora.enabled === null ? "Status not recorded · " : ""}
      Model weight: {lora.modelStrength ?? "not recorded"}
      {" · "}CLIP weight: {lora.clipStrength ?? "not recorded"}
    </small>
  </li>;
}

export function RecordedGenerationConfiguration({ provenance }: { provenance: unknown }) {
  const details = recordedGenerationDetails(provenance);
  return <div className="generation-details-content">
    <GenerationIdentitySummary identity={details.identity} />
    {!details.identity && <p>Model and workflow names were not recorded.</p>}
    {details.settings.length > 0 ? <dl className="generation-identity">
      {details.settings.map(({ label, value }) => <Fragment key={label}>
        <dt>{label}</dt><dd>{value}</dd>
      </Fragment>)}
    </dl> : <p>Generation settings were not recorded.</p>}
    <strong>Added LoRAs</strong>
    {details.addedLoras.length > 0 ? <ul>
      {details.addedLoras.map((lora, index) => <LoraDetails key={index} lora={lora} />)}
    </ul> : <p>{details.addedLorasRecorded ? "No added LoRAs." : "Added LoRAs were not recorded."}</p>}
    {details.workflowLoras.length > 0 && <>
      <strong>Workflow LoRA overrides</strong>
      <ul>{details.workflowLoras.map((lora, index) => <LoraDetails key={index} lora={lora} />)}</ul>
    </>}
    <p>Shows the recorded generation. Other workflow-embedded LoRAs may not be recorded.</p>
  </div>;
}

export function GenerationDetails({ provenance }: { provenance: unknown }) {
  const [open, setOpen] = useState(false);
  return <details className="generation-details" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Generation details</summary>
    {open && <RecordedGenerationConfiguration provenance={provenance} />}
  </details>;
}
