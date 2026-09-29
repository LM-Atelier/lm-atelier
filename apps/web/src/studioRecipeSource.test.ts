import { describe, expect, it } from "vitest";
import { studioRecipeSource } from "./studioRecipeSource";
import type { ChatDetail, Message, MessagePart } from "./types";
import type { StudioStep } from "./useStudioSession";

const stamp = "2026-09-29T00:00:00Z";

function answer(id: string, metadata: Record<string, unknown> | null): Message {
  const parts: MessagePart[] = [
    { id: `${id}-image`, position: 0, type: "image", text: null, artifact_id: `art-${id}`, metadata_json: {} },
  ];
  if (metadata) {
    parts.push({ id: `${id}-meta`, position: 1, type: "generation_metadata", text: null, artifact_id: null, metadata_json: metadata });
  }
  return { id, chat_id: "chat-studio", parent_id: null, role: "assistant", status: "complete", parts, created_at: stamp, updated_at: stamp };
}

function session(messages: Message[]): ChatDetail {
  return { messages } as unknown as ChatDetail;
}

function step(messageId: string, overrides: Partial<StudioStep> = {}): StudioStep {
  return {
    messageId,
    artifactId: `art-${messageId}`,
    instruction: "make it a watercolor",
    beforeArtifactId: "art-source",
    isSource: false,
    generationIdentity: null,
    ...overrides,
  };
}

describe("the recipe a Studio result can be saved as", () => {
  it("is the run that made it and the words it was given", () => {
    const detail = session([answer("made", { run_id: "run-7", provenance: {} })]);

    expect(studioRecipeSource(detail, step("made"))).toEqual({ runId: "run-7", instruction: "make it a watercolor" });
  });

  it("is nothing for an exact edit, which records no run", () => {
    const detail = session([answer("turned", { provenance: { local_edit: { operation: "rotate_clockwise" } } })]);

    expect(studioRecipeSource(detail, step("turned", { instruction: "Rotate right" }))).toBeNull();
  });

  it("is nothing for the original picture, or with nothing chosen", () => {
    const detail = session([answer("made", { run_id: "run-7" })]);

    expect(studioRecipeSource(detail, step("source", { isSource: true, instruction: "" }))).toBeNull();
    expect(studioRecipeSource(detail, null)).toBeNull();
  });

  it("is nothing for a result recorded without words, which a recipe could not keep", () => {
    const detail = session([answer("made", { run_id: "run-7" })]);

    expect(studioRecipeSource(detail, step("made", { instruction: "  " }))).toBeNull();
  });
});
