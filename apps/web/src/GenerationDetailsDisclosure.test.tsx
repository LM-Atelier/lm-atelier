import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MessageBubble } from "./MessageBubble";
import { MediaLibraryView } from "./MediaLibraryView";
import { api } from "./api";
import { parseArtifactLibraryPage } from "./artifactLibraryPage";
import type { Message, Run, Artifact } from "./types";

vi.mock("./api", () => ({ api: { artifactLibrary: vi.fn(), artifact: vi.fn(), run: vi.fn() } }));
afterEach(() => { cleanup(); vi.resetAllMocks(); });
const artifactId = `sha256:${"a".repeat(64)}`;
const stamp = "2026-09-30T12:00:00Z";
const provenance = {
  model: { profile_name: "Captured model" },
  workflow: { family_name: "Captured workflow", version: 4 },
  resolved_settings: { seed: 17, width: 512, frames: 81, fps: 24 },
  auxiliary_assets: { lora_stack: [{ name: "Watercolor", enabled: true, model_strength: 0.7, clip_strength: 0.2 }] },
};
function artifact(): Artifact {
  return {
    id: artifactId, sha256: "a".repeat(64), kind: "image", media_type: "image/png",
    size_bytes: 1024, original_name: "study.png", metadata_json: { run_id: "run" }, created_at: stamp,
  };
}
function message(kind: "image" | "video" = "image", record: unknown = provenance): Message {
  return {
    id: "message", chat_id: "chat", parent_id: null, role: "assistant", status: "complete",
    transcript_visible: true, content_removed_at: null, active_response_revision_id: null,
    parts: [
      { id: "output", position: 0, type: kind, text: null, artifact_id: artifactId, metadata_json: {} },
      { id: "metadata", position: 1, type: "generation_metadata", text: null, artifact_id: null, metadata_json: { provenance: record } },
    ], references: [], response_revisions: [], feedback: null, created_at: stamp, updated_at: stamp,
  };
}
async function openDetails() {
  const summary = screen.getByText("Generation details", { selector: "summary" });
  const details = summary.parentElement as HTMLDetailsElement;
  expect(details.open).toBe(false);
  details.open = true;
  fireEvent(details, new Event("toggle"));
  await waitFor(() => expect(summary.parentElement?.querySelector(".generation-details-content")).not.toBeNull());
  return { summary, details };
}
function library() {
  vi.mocked(api.artifactLibrary).mockResolvedValue(parseArtifactLibraryPage({ items: [{
    id: `libentry:${artifactId}`, artifact_id: artifactId, version: 1, state: "visible",
    display_name: "Watercolor study", favorite: false, kind: "image", media_type: "image/png", size_bytes: 1024,
    created_at: stamp, updated_at: stamp,
  }], next_cursor: null }, 20));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><MediaLibraryView /></QueryClientProvider>);
}

describe("generation details beside outputs", () => {
  it.each(["image", "video"] as const)("discloses the %s's own record only after opening", async (kind) => {
    const { rerender } = render(<MessageBubble message={message(kind)} />);
    expect(screen.queryByText("Captured model")).toBeNull();
    const { summary, details } = await openDetails();
    expect(screen.getByText("Captured model")).toBeVisible();
    expect(screen.getByText("Watercolor")).toBeVisible();
    expect(screen.getByText("Model weight: 0.7 · CLIP weight: 0.2")).toBeVisible();
    summary.focus();
    const changed = message(kind, { ...provenance, model: { profile_name: "Different recorded revision" } });
    rerender(<MessageBubble message={changed} />);
    expect(screen.queryByText("Captured model")).toBeNull();
    expect(screen.getByText("Different recorded revision")).toBeVisible();
    expect(summary).toHaveFocus();
    details.open = false;
    fireEvent(details, new Event("toggle"));
    await waitFor(() => expect(screen.queryByText("Watercolor")).toBeNull());
    expect(api.artifact).not.toHaveBeenCalled();
    expect(api.run).not.toHaveBeenCalled();
  });

  it("states missing historical configuration without substituting current defaults", async () => {
    render(<MessageBubble message={message("image", null)} />);
    await openDetails();
    expect(screen.getByText("Generation settings were not recorded.")).toBeVisible();
    expect(screen.getByText("Added LoRAs were not recorded.")).toBeVisible();
  });

  it("keeps previews, uploaded inputs and removed responses free of generation controls", () => {
    const input = message();
    input.role = "user";
    input.parts[0]!.metadata_json = { input_reference: true };
    const { rerender } = render(<MessageBubble message={input} />);
    expect(screen.queryByText("Generation details")).toBeNull();
    const preview = message();
    preview.parts[0]!.metadata_json = { preview: true };
    rerender(<MessageBubble message={preview} />);
    expect(screen.queryByText("Generation details")).toBeNull();
    rerender(<MessageBubble message={{ ...message(), content_removed_at: stamp }} />);
    expect(screen.queryByText("Generation details")).toBeNull();
  });

  it("reads one library generation lazily and refuses a run that does not contain that output", async () => {
    vi.mocked(api.artifact).mockResolvedValue(artifact());
    vi.mocked(api.run).mockResolvedValue({ provenance_json: { ...provenance, outputs: [{ artifact_id: "different artifact" }] } } as unknown as Run);
    library();
    await screen.findByText("Watercolor study");
    expect(api.artifact).not.toHaveBeenCalled();
    expect(api.run).not.toHaveBeenCalled();
    await openDetails();
    expect(api.artifact).toHaveBeenCalledExactlyOnceWith(artifactId);
    expect(api.run).toHaveBeenCalledExactlyOnceWith("run");
    expect(screen.queryByText("Captured model")).toBeNull();
    expect(screen.getByText("Model and workflow names were not recorded.")).toBeVisible();
  });

  it("shows library provenance and keeps read failures local to the opened details", async () => {
    vi.mocked(api.artifact).mockResolvedValue(artifact());
    vi.mocked(api.run).mockResolvedValue({ provenance_json: { ...provenance, outputs: [{ artifact_id: artifactId }] } } as unknown as Run);
    library();
    await screen.findByText("Watercolor study");
    await openDetails();
    expect(screen.getByText("Captured model")).toBeVisible();
  });

  it("does not display raw API errors or fill missing details after a failed read", async () => {
    vi.mocked(api.artifact).mockRejectedValue(new Error("neutral private error marker"));
    library();
    await screen.findByText("Watercolor study");
    const details = screen.getByText("Generation details", { selector: "summary" }).parentElement as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    expect(await screen.findByRole("alert")).toHaveTextContent("Generation details could not be loaded.");
    expect(screen.queryByText("neutral private error marker")).toBeNull();
    expect(api.run).not.toHaveBeenCalled();
  });
});
