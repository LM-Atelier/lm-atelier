import { useState } from "react";
import { LibraryImagePicker } from "./LibraryImagePicker";
import { artifactSource } from "./messageMedia";
import { ShieldedThumbnail } from "./ShieldedThumbnail";
import type { ComparisonSource } from "./generationComparison";
import "./GenerationComparisonView.css";

/** What both choices start from: words alone, or one picture from the Media Library to change. */
export function ComparisonPictureField({ value, onChange }: {
  value: ComparisonSource | null;
  onChange: (next: ComparisonSource | null) => void;
}) {
  const [choosing, setChoosing] = useState(false);
  return <fieldset className="comparison-shared-group">
    <legend>Start from</legend>
    <label className="comparison-radio"><input type="radio" name="comparison-start" checked={value === null}
      onChange={() => onChange(null)} />Words only: make a new picture</label>
    <label className="comparison-radio"><input type="radio" name="comparison-start" checked={value !== null}
      onChange={() => { if (value === null) setChoosing(true); }} />A picture from the Media Library: change it</label>
    {value && <div className="comparison-start-picture">
      <ShieldedThumbnail src={artifactSource(value.id) ?? undefined} kind="image" />
      <p>Both choices change this picture and keep its size.</p>
      <button type="button" className="secondary" onClick={() => setChoosing(true)}>Choose another picture</button>
    </div>}
    {choosing && <LibraryImagePicker title="The picture both choices change" confirmLabel="Change this picture" single
      onConfirm={(items) => {
        const [item] = items;
        if (item) onChange({ id: item.id });
      }}
      onClose={() => setChoosing(false)} />}
  </fieldset>;
}
