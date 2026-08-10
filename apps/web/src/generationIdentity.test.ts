import { describe, expect, it } from "vitest";
import { generationIdentityFromProvenance } from "./generationIdentity";

describe("captured generation identity", () => {
  it("projects only bounded display names and a stable workflow version", () => {
    expect(generationIdentityFromProvenance({
      model: { profile_name: " Krea 2 edit ", local_path: "private/model" },
      workflow: {
        family_name: "Krea 2 edits",
        definition_name: "Krea 2 inpaint",
        version: 7,
        dependencies: { private: true },
      },
      prompt: "private prompt",
    })).toEqual({
      model_profile_name: "Krea 2 edit",
      workflow_family_name: "Krea 2 edits",
      workflow_definition_name: "Krea 2 inpaint",
      workflow_version: 7,
    });
  });

  it("does not make a label from malformed legacy provenance", () => {
    expect(generationIdentityFromProvenance({
      model: { profile_name: "x".repeat(501) },
      workflow: { family_name: [], definition_name: false, version: 7 },
    })).toBeNull();
    expect(generationIdentityFromProvenance(null)).toBeNull();
  });
});
