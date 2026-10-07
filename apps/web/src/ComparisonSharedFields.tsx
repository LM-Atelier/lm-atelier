import { RATIO_LABELS } from "./outputRatio";
import { SEED_POLICY_LABELS, type ComparisonDraft } from "./generationComparison";
import type { SeedPolicyKind } from "./generationExperimentTypes";
import type { OutputRatioPresetId } from "./types";
import "./GenerationComparisonView.css";

const SEED_KINDS: SeedPolicyKind[] = ["same_recorded_number", "fixed_numeric", "independent_deterministic", "random_per_trial"];

/** What both choices share: the words, the size, a video's length and how their seeds are chosen. */
export function ComparisonSharedFields({ value, onChange, sharedPresets, shapesKnown }: {
  value: ComparisonDraft;
  onChange: (next: ComparisonDraft) => void;
  sharedPresets: OutputRatioPresetId[];
  shapesKnown: boolean;
}) {
  const size = value.size;
  return <div className="comparison-shared">
    <label>Prompt<textarea value={value.prompt} maxLength={200_000} rows={4}
      onChange={(event) => onChange({ ...value, prompt: event.target.value })} /></label>
    <label>Negative prompt<textarea value={value.negativePrompt} maxLength={20_000} rows={2}
      onChange={(event) => onChange({ ...value, negativePrompt: event.target.value })} /></label>
    {value.source ? <p className="comparison-note">Size: the picture's own, for both choices.</p> : <fieldset className="comparison-shared-group">
      <legend>Size</legend>
      <label className="comparison-radio"><input type="radio" name="comparison-size" checked={size.mode === "size"}
        onChange={() => onChange({ ...value, size: { mode: "size", width: "1024", height: "1024" } })} />Exact size</label>
      {size.mode === "size" && <div className="comparison-inline">
        <label>Width<input inputMode="numeric" value={size.width}
          onChange={(event) => onChange({ ...value, size: { ...size, width: event.target.value } })} /></label>
        <label>Height<input inputMode="numeric" value={size.height}
          onChange={(event) => onChange({ ...value, size: { ...size, height: event.target.value } })} /></label>
      </div>}
      <label className="comparison-radio"><input type="radio" name="comparison-size" checked={size.mode === "preset"}
        onChange={() => onChange({ ...value, size: { mode: "preset", presetId: "" } })} />Shape</label>
      {size.mode === "preset" && <label>Shape both workflows can make<select aria-label="Shape" value={size.presetId}
        onChange={(event) => onChange({ ...value, size: { mode: "preset", presetId: event.target.value as OutputRatioPresetId | "" } })}>
        <option value="">{shapesKnown ? (sharedPresets.length ? "Choose a shape" : "No shape both workflows can make") : "Choose both workflows first"}</option>
        {sharedPresets.map((presetId) => <option key={presetId} value={presetId}>{`${RATIO_LABELS[presetId]} (${presetId})`}</option>)}
      </select></label>}
    </fieldset>}
    {value.video && !value.source && <label>Length in seconds (optional)
      <input inputMode="decimal" value={value.seconds}
        onChange={(event) => onChange({ ...value, seconds: event.target.value })} />
      <small>Each workflow makes the length nearest to this that it can; left empty, each makes its own.</small>
    </label>}
    <fieldset className="comparison-shared-group">
      <legend>Seed</legend>
      {SEED_KINDS.map((kind) => <label key={kind} className="comparison-radio"><input type="radio" name="comparison-seed"
        checked={value.seed.kind === kind} onChange={() => onChange({ ...value, seed: { ...value.seed, kind } })} />{SEED_POLICY_LABELS[kind]}</label>)}
      {value.seed.kind !== "random_per_trial" && <label>
        {value.seed.kind === "fixed_numeric" ? "Seed number" : "Seed number (optional)"}
        <input inputMode="numeric" value={value.seed.number}
          onChange={(event) => onChange({ ...value, seed: { ...value.seed, number: event.target.value } })} />
      </label>}
    </fieldset>
    {/* A blind viewing shows each result as a picture made anew, which a video cannot be. */}
    {!(value.video && !value.source) && <label className="comparison-radio"><input type="checkbox" checked={value.blind}
      onChange={(event) => onChange({ ...value, blind: event.target.checked })} />
      Compare blind: hide which choice made each picture until you say which you prefer</label>}
  </div>;
}
