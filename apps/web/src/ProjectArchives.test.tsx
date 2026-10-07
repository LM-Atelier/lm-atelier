/** Exporting and importing project archives from Data & backups. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
  expect(api.projects).toHaveBeenCalledWith(true, "", expect.objectContaining({ limit: 50, offset: 0, literalSearch: true }));
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

it("exports encrypted only once the passphrase is typed the same way twice", async () => {
  vi.mocked(api.exportProject).mockResolvedValue({ url: "/api/artifacts/archive-2/content" });
  show();
  await screen.findByText("Garden");

  fireEvent.click(screen.getByLabelText(/Encrypt exports with a passphrase/));
  fireEvent.change(screen.getByLabelText("Passphrase"), { target: { value: "correct horse" } });
  fireEvent.change(screen.getByLabelText("Confirm passphrase"), { target: { value: "correct hose" } });
  expect(screen.getByText("The passphrases do not match.")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Export Garden with media" }));
  expect(api.exportProject).not.toHaveBeenCalled();

  fireEvent.change(screen.getByLabelText("Confirm passphrase"), { target: { value: "correct horse" } });
  fireEvent.click(screen.getByRole("button", { name: "Export Garden with media" }));
  await waitFor(() => expect(api.exportProject).toHaveBeenLastCalledWith("p-1", true, "correct horse"));
  await waitFor(() => expect(downloads).toEqual(["/api/artifacts/archive-2/content"]));
});

const ENCRYPTED_HEAD = [0x4c, 0x4d, 0x41, 0x41, 0x52, 0x43, 0x48, 0x00];

it("asks for an encrypted archive's passphrase before sending it", async () => {
  vi.mocked(api.importProject).mockResolvedValue(project("p-3", "Travel"));
  show();
  await screen.findByText("Garden");

  const encrypted = new File([new Uint8Array([...ENCRYPTED_HEAD, 1, 2, 3])], "travel.lm-atelier.encrypted");
  fireEvent.change(screen.getByLabelText("Project archive to import"), { target: { files: [encrypted] } });
  expect(await screen.findByText("travel.lm-atelier.encrypted is encrypted. Enter its passphrase to import it.")).toBeTruthy();
  expect(api.importProject).not.toHaveBeenCalled();

  fireEvent.change(screen.getByLabelText("Archive passphrase"), { target: { value: "correct horse" } });
  fireEvent.click(screen.getByRole("button", { name: "Import" }));
  await waitFor(() => expect(api.importProject).toHaveBeenCalledWith(encrypted, "correct horse"));
  expect(await screen.findByRole("status")).toHaveTextContent("Imported Travel.");
  expect(screen.queryByLabelText("Archive passphrase")).toBeNull();
});

it("keeps asking after a passphrase that does not open the archive", async () => {
  vi.mocked(api.importProject).mockRejectedValue(new Error("The passphrase is wrong, or the archive is damaged."));
  show();
  await screen.findByText("Garden");

  const encrypted = new File([new Uint8Array(ENCRYPTED_HEAD)], "travel.lm-atelier.encrypted");
  fireEvent.change(screen.getByLabelText("Project archive to import"), { target: { files: [encrypted] } });
  fireEvent.change(await screen.findByLabelText("Archive passphrase"), { target: { value: "wrong" } });
  fireEvent.click(screen.getByRole("button", { name: "Import" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("The passphrase is wrong, or the archive is damaged.");
  expect(screen.getByLabelText("Archive passphrase")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.queryByLabelText("Archive passphrase")).toBeNull();
});

it("puts the passphrase prompt in focus with its own error, and clears both when it closes", async () => {
  vi.mocked(api.importProject)
    .mockRejectedValueOnce(new Error("The passphrase is wrong, or the archive is damaged."))
    .mockResolvedValue(project("p-4", "Plain"));
  show();
  await screen.findByText("Garden");
  const encrypted = new File([new Uint8Array(ENCRYPTED_HEAD)], "travel.lm-atelier.encrypted");
  const picker = screen.getByLabelText("Project archive to import");

  fireEvent.change(picker, { target: { files: [encrypted] } });
  const field = await screen.findByLabelText("Archive passphrase");
  await waitFor(() => expect(field).toHaveFocus());
  fireEvent.change(field, { target: { value: "wrong" } });
  fireEvent.click(screen.getByRole("button", { name: "Import" }));
  const prompt = screen.getByRole("form", { name: "Encrypted archive" });
  expect(await within(prompt).findByRole("alert")).toHaveTextContent("The passphrase is wrong, or the archive is damaged.");

  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("alert")).toBeNull();
  fireEvent.change(picker, { target: { files: [encrypted] } });
  expect(await screen.findByLabelText("Archive passphrase")).toHaveValue("");
  expect(screen.queryByRole("alert")).toBeNull();

  const plain = new File([new Uint8Array([0x50, 0x4b, 3, 4])], "plain.lm-atelier.zip");
  fireEvent.change(picker, { target: { files: [plain] } });
  await waitFor(() => expect(api.importProject).toHaveBeenLastCalledWith(plain));
  expect(screen.queryByLabelText("Archive passphrase")).toBeNull();
});

it("says why the export buttons wait while encryption is on", async () => {
  show();
  await screen.findByText("Garden");

  fireEvent.click(screen.getByLabelText(/Encrypt exports with a passphrase/));

  expect(screen.getByRole("button", { name: "Export Garden with media" })).toHaveAccessibleDescription(
    "Type the passphrase twice to export encrypted.",
  );
  fireEvent.change(screen.getByLabelText("Passphrase"), { target: { value: "correct horse" } });
  fireEvent.change(screen.getByLabelText("Confirm passphrase"), { target: { value: "correct horse" } });
  expect(screen.getByRole("button", { name: "Export Garden with media" })).not.toHaveAccessibleDescription();
});
