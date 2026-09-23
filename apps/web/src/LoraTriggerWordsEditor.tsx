import { useState } from "react";
import { measuredTriggerWords, parseTypedTriggerWords } from "./loraTriggerWords";
import type { ModelAssetInstall } from "./types";

/** Record the trigger words a LoRA needs that its own file does not declare.
 *
 * The file's words are shown in full as a record of what was measured, and
 * are not editable. What a person types is kept as their own and added to the
 * prompt the same way whenever the LoRA is used.
 */
export function LoraTriggerWordsEditor({
  asset,
  saving,
  onSave,
  onCancel,
}: {
  asset: Pick<ModelAssetInstall, "name" | "manifest_json" | "typed_trigger_words">;
  saving: boolean;
  onSave: (typedTriggerWords: string[]) => void;
  onCancel: () => void;
}) {
  const [text, setText] = useState(asset.typed_trigger_words.join(", "));
  const measured = measuredTriggerWords(asset);
  const typed = parseTypedTriggerWords(text);
  const unchanged = typed.join("\n") === asset.typed_trigger_words.join("\n");
  return (
    <form
      className="model-use-case-editor lora-trigger-editor"
      onSubmit={(event) => {
        event.preventDefault();
        if (!unchanged) onSave(typed);
      }}
    >
      <small>
        {measured.length > 0
          ? `From the file: ${measured.join(", ")}`
          : "The file declares no trigger words."}
      </small>
      <label>
        Your trigger words
        <input
          aria-label={`Trigger words for ${asset.name}`}
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Separate words with commas"
        />
      </label>
      <span className="row-actions">
        <button type="button" className="secondary compact-button" disabled={saving} onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" className="primary compact-button" disabled={saving || unchanged}>
          {saving ? "Saving…" : "Save"}
        </button>
      </span>
      <small>Added to the prompt whenever this LoRA is used.</small>
    </form>
  );
}
