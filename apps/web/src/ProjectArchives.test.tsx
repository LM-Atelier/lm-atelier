/** Exporting and importing project archives from Data & backups. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ProjectArchives } from "./ProjectArchives";
import type { Project } from "./types";

vi.mock("./api", () => ({ api: { projects: vi.fn(), exportProject: vi.fn(), importProject: vi.fn() } }));

function project(id: string, name: string, archived = false): Project {
  return {
    id, name, description: "", instructions: "", archived, pinned: false,
    image_workflow_revision_id: null, video_workflow_revision_id: null,
    created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-01T00:00:00Z",
  };
}

const downloads: string[] = [];

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><ProjectArchives /></QueryClientProvider>);
}

beforeEach(() => {
  downloads.length = 0;
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push(this.getAttribute("href") ?? "");
  });
  vi.mocked(api.projects).mockResolvedValue([project("p-1", "Garden"), project("p-2", "Old notes", true)]);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("lists every project, archived ones too, and exports with or without media", async () => {
  vi.mocked(api.exportProject).mockResolvedValue({ url: "/api/artifacts/archive-1/content" });
  show();

  expect(await screen.findByText("Old notes")).toBeTruthy();
  expect(api.projects).toHaveBeenCalledWith(true);
  expect(screen.getByText("Archived")).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Export Garden with media" }));
  await waitFor(() => expect(downloads).toEqual(["/api/artifacts/archive-1/content"]));
  expect(api.exportProject).toHaveBeenLastCalledWith("p-1", true);

  fireEvent.click(screen.getByRole("button", { name: "Export Old notes, metadata only" }));
  await waitFor(() => expect(api.exportProject).toHaveBeenLastCalledWith("p-2", false));
});

it("imports an archive, says what arrived, and lists it", async () => {
  vi.mocked(api.importProject).mockImplementation(async () => {
    vi.mocked(api.projects).mockResolvedValue([project("p-1", "Garden"), project("p-2", "Old notes", true), project("p-3", "Travel")]);
    return project("p-3", "Travel");
  });
  show();
  await screen.findByText("Garden");

  const archive = new File(["zip"], "travel.lm-atelier.zip", { type: "application/zip" });
  fireEvent.change(screen.getByLabelText("Project archive to import"), { target: { files: [archive] } });

  expect(await screen.findByRole("status")).toHaveTextContent("Imported Travel.");
  expect(vi.mocked(api.importProject).mock.calls[0]?.[0]).toBe(archive);
  expect(await screen.findByText("Travel", { selector: "strong" })).toBeTruthy();
});

it("says there is nothing to export when there are no projects", async () => {
  vi.mocked(api.projects).mockResolvedValue([]);
  show();

  expect(await screen.findByText("No projects to export yet.")).toBeTruthy();
});

it("reports an archive that could not be imported", async () => {
  vi.mocked(api.importProject).mockRejectedValue(new Error("This file is not a project archive."));
  show();
  await screen.findByText("Garden");

  fireEvent.change(screen.getByLabelText("Project archive to import"), {
    target: { files: [new File(["x"], "notes.zip", { type: "application/zip" })] },
  });

  expect(await screen.findByRole("alert")).toHaveTextContent("This file is not a project archive.");
  expect(screen.queryByRole("status")).toBeNull();
});
