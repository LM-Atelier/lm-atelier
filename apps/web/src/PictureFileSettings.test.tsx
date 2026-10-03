import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { PictureFileSettings } from "./PictureFileSettings";
import { MediaLibraryView } from "./MediaLibraryView";
import { api } from "./api";
import { parseArtifactLibraryPage } from "./artifactLibraryPage";

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: { artifactLibrary: vi.fn(), pictureSettings: vi.fn() },
}));
afterEach(() => { cleanup(); vi.resetAllMocks(); });

const artifactId = `sha256:${"b".repeat(64)}`;
const stamp = "2026-10-03T12:00:00Z";
const answer = {
  dialect: "parameters",
  parser_version: 1,
  budget_version: 1,
  digest: `sha256:${"c".repeat(64)}`,
  claims: [
    { key: "prompt", value: "a ceramic cup on a wooden table", source: "prompt" },
    { key: "negative_prompt", value: "blurry", source: "Negative prompt" },
    // A seed past what a number holds exactly arrives as its decimal text.
    { key: "seed", value: "987654321012345678", source: "Seed" },
    { key: "guidance", value: 6.5, source: "CFG scale" },
  ],
  ignored: [
    { name: "Model", reason: "names_a_file" },
    { name: "workflow", reason: "workflow_graph" },
    { name: "Version", reason: "something_new" },
    { name: "KSampler.negative", reason: "empty" },
  ],
  warnings: [],
};

function renderSettings() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><PictureFileSettings artifactId={artifactId} pictureName="Ceramic cup" /></QueryClientProvider>);
}

function open() {
  const summary = screen.getByText("Settings in the file", { selector: "summary" });
  const details = summary.parentElement as HTMLDetailsElement;
  details.open = true;
  fireEvent(details, new Event("toggle"));
}

describe("settings stored in a picture's file", () => {
  it("reads the file only when opened and lists what it says and what was left out", async () => {
    vi.mocked(api.pictureSettings).mockResolvedValue(answer);
    renderSettings();
    expect(api.pictureSettings).not.toHaveBeenCalled();

    open();

    expect(await screen.findByText("a ceramic cup on a wooden table")).toBeVisible();
    expect(api.pictureSettings).toHaveBeenCalledWith(artifactId, expect.any(AbortSignal));
    const terms = screen.getAllByRole("term").map((term) => term.textContent);
    expect(terms).toEqual(["Prompt", "Negative prompt", "Seed", "Guidance"]);
    expect(screen.getByText("987654321012345678")).toBeVisible();
    expect(screen.getByText("6.5")).toBeVisible();
    const leftOut = screen.getAllByRole("listitem").map((item) => item.textContent);
    expect(leftOut).toEqual([
      "Model: names a file on the computer that made it",
      "workflow: a workflow, which is never run from here",
      "Version: left out",
      "KSampler.negative: empty",
    ]);
    expect(screen.queryByText("More was left out than is listed here.")).toBeNull();
    expect(screen.getByText(/Nothing named in it is fetched, run or kept/)).toBeVisible();
  });

  it.each([
    [["format_not_read"], "Only PNG, JPEG and WebP pictures are read for settings."],
    [["too_large"], "This picture is too large to read for settings."],
    [["no_settings_found"], "No settings were found in this picture's file."],
    [["several_samplers"], "This picture's workflow has more than one sampler, so no one set of settings is shown."],
    [["no_sampler"], "This picture's workflow has no sampler to read settings from."],
    [["settings_unrecognized"], "The settings text in this picture has no line of settings."],
  ])("says why a file with warnings %j shows no settings", async (warnings, note) => {
    vi.mocked(api.pictureSettings).mockResolvedValue({
      ...answer, dialect: "none", digest: null, claims: [], ignored: [], warnings,
    });
    renderSettings();
    open();

    expect(await screen.findByText(note)).toBeVisible();
    expect(screen.queryByRole("term")).toBeNull();
    expect(screen.queryByText("Left out")).toBeNull();
  });

  it("says settings were found but none could be read, and when the list was cut short", async () => {
    vi.mocked(api.pictureSettings).mockResolvedValue({
      ...answer, dialect: "comfyui_prompt", claims: [], warnings: ["too_many_entries"],
    });
    renderSettings();
    open();

    expect(await screen.findByText("No settings could be read from this picture's file.")).toBeVisible();
    expect(screen.getByText("More was left out than is listed here.")).toBeVisible();
  });

  it.each([
    [{ code: "generation-settings-unreadable" }, "The settings text stored in this picture could not be read."],
    [{ code: "artifact-file-unreadable" }, "This picture's file could not be read."],
  ])("explains a failed reading %j", async (failure, message) => {
    vi.mocked(api.pictureSettings).mockRejectedValue(Object.assign(new Error("failed"), failure));
    renderSettings();
    open();

    expect(await screen.findByRole("alert")).toHaveTextContent(message);
  });

  it.each([
    { ...answer, dialect: "something_else" },
    { ...answer, claims: [{ key: "seed", value: { nested: true }, source: "Seed" }] },
    { ...answer, ignored: "Model" },
    { ...answer, warnings: [1] },
    // A whole number past what a number holds exactly has already been rounded.
    { ...answer, claims: [{ key: "seed", value: 2 ** 60, source: "Seed" }] },
  ])("shows nothing from an answer of another shape", async (other) => {
    vi.mocked(api.pictureSettings).mockResolvedValue(other);
    renderSettings();
    open();

    expect(await screen.findByRole("alert")).toHaveTextContent("This picture's file could not be read.");
    expect(screen.queryByRole("term")).toBeNull();
  });

  it("is offered for pictures in the Media Library and not for videos", async () => {
    const entry = (id: string, kind: "image" | "video", name: string) => ({
      id: `libentry:${id}`, artifact_id: id, version: 1, state: "visible", display_name: name,
      favorite: false, kind, media_type: kind === "image" ? "image/png" : "video/mp4", size_bytes: 1024,
      created_at: stamp, updated_at: stamp,
    });
    // The library lists newest first, then by id, highest first.
    vi.mocked(api.artifactLibrary).mockResolvedValue(parseArtifactLibraryPage({ items: [
      entry(`sha256:${"d".repeat(64)}`, "video", "Turntable clip"),
      entry(artifactId, "image", "Ceramic study"),
    ], next_cursor: null }, 20));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><MediaLibraryView /></QueryClientProvider>);

    const picture = (await screen.findByText("Ceramic study")).closest("article") as HTMLElement;
    const clip = screen.getByText("Turntable clip").closest("article") as HTMLElement;
    expect(within(picture).getByText("Settings in the file")).toBeVisible();
    expect(within(clip).queryByText("Settings in the file")).toBeNull();
    await waitFor(() => expect(api.pictureSettings).not.toHaveBeenCalled());
  });
});
