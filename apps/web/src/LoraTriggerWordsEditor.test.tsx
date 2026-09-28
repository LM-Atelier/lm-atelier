import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LoraTriggerWordsEditor } from "./LoraTriggerWordsEditor";
import { loraTriggerWords, measuredTriggerWords, parseTypedTriggerWords } from "./loraTriggerWords";

afterEach(cleanup);

const declared = {
  name: "Atelier Ink",
  manifest_json: { metadata: { trigger_words: ["ink wash"], trained_words: ["Ink Wash", "soft edge"] } },
  typed_trigger_words: ["studio glow", "SOFT EDGE"],
};

describe("LoRA trigger words", () => {
  it("reads the file's words the way the server does, then the typed ones", () => {
    expect(measuredTriggerWords(declared)).toEqual(["ink wash", "soft edge"]);
    // A typed word the file already declares, in any casing, is not added twice.
    expect(loraTriggerWords(declared)).toEqual(["ink wash", "soft edge", "studio glow"]);
    expect(measuredTriggerWords({ manifest_json: {} })).toEqual([]);
  });

  it("reads what a person typed as words separated by commas", () => {
    expect(parseTypedTriggerWords(" studio glow, ,Studio Glow,soft edge ")).toEqual([
      "studio glow",
      "soft edge",
    ]);
    expect(parseTypedTriggerWords("")).toEqual([]);
  });

  it("shows the file's words as the file's, and saves only the typed ones", () => {
    const onSave = vi.fn();
    render(
      <LoraTriggerWordsEditor
        asset={{ ...declared, typed_trigger_words: [] }}
        saving={false}
        onSave={onSave}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText("From the file: ink wash, soft edge")).toBeInTheDocument();
    expect(screen.getByLabelText("Trigger words for Atelier Ink")).toHaveValue("");
    const save = screen.getByRole("button", { name: "Save" });
    expect(save).toBeDisabled();

    fireEvent.change(screen.getByLabelText("Trigger words for Atelier Ink"), {
      target: { value: "studio glow, watercolor edge" },
    });
    fireEvent.click(save);

    expect(onSave).toHaveBeenCalledWith(["studio glow", "watercolor edge"]);
  });

  it("says so when the file declares nothing", () => {
    render(
      <LoraTriggerWordsEditor
        asset={{ name: "Plain", manifest_json: {}, typed_trigger_words: ["studio glow"] }}
        saving={false}
        onSave={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText("The file declares no trigger words.")).toBeInTheDocument();
    expect(screen.getByLabelText("Trigger words for Plain")).toHaveValue("studio glow");
  });
});
