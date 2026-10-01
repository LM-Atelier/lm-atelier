import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ProjectPicker } from "./ProjectPicker";
import type { Project } from "./types";

vi.mock("./api", () => ({ api: { projects: vi.fn(), project: vi.fn() } }));

const project: Project = {
  id: "selected", name: "Saved notebook", description: "", instructions: "",
  archived: false, pinned: false, created_at: "2026-09-30", updated_at: "2026-09-30",
  image_workflow_revision_id: null, video_workflow_revision_id: null,
};
const clients: QueryClient[] = [];

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  const change = vi.fn();
  render(<QueryClientProvider client={client}>
    <ProjectPicker value={project.id} onChange={change} />
  </QueryClientProvider>);
  return { client, change };
}

beforeEach(() => {
  vi.mocked(api.projects).mockResolvedValue([project]);
  vi.mocked(api.project).mockResolvedValue(project);
});
afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); vi.resetAllMocks(); });

it("keeps a pending selected project distinct from a cached page choice", async () => {
  let finish!: (value: Project) => void;
  vi.mocked(api.project).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  const { change } = show();
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Project" })).toBeEnabled());
  expect(await screen.findByRole("option", { name: "Loading selected project…" })).toBeInTheDocument();
  expect(screen.queryByRole("option", { name: "Saved notebook" })).toBeNull();
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id);
  finish({ ...project, name: "Current notebook" });
  await screen.findByRole("option", { name: "Current notebook" });
  expect(change).not.toHaveBeenCalled();
});

it("uses exact selected metadata instead of an older page name or archive state", async () => {
  vi.mocked(api.project).mockResolvedValue({ ...project, name: "Current notebook", archived: true });
  show();
  await waitFor(() => expect(api.project).toHaveBeenCalledWith(project.id, expect.any(AbortSignal)));
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Project" })).toBeEnabled());
  expect(await screen.findByRole("option", { name: "Current notebook (Archived)" })).toBeInTheDocument();
  expect(screen.queryByRole("option", { name: "Saved notebook" })).toBeNull();
  expect(screen.getAllByRole("option").filter(option => (option as HTMLOptionElement).value === project.id)).toHaveLength(1);
});

it("keeps a failed selected-project read neutral and recovers its label on retry", async () => {
  vi.mocked(api.project).mockRejectedValueOnce(new Error("Selected project read failed"))
    .mockResolvedValue({ ...project, name: "Current notebook" });
  const { change } = show();
  await screen.findByText("Selected project read failed");
  expect(screen.getByRole("option", { name: "Selected project" })).toBeInTheDocument();
  expect(screen.queryByRole("option", { name: /unavailable/ })).toBeNull();
  expect(screen.queryByRole("option", { name: "Saved notebook" })).toBeNull();
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id);
  fireEvent.click(screen.getByRole("button", { name: "Retry selected project" }));
  await screen.findByRole("option", { name: "Current notebook" });
  expect(change).not.toHaveBeenCalled();
});

it("does not restore cached selected metadata when a refresh fails", async () => {
  const { client, change } = show();
  await screen.findByRole("option", { name: "Saved notebook" });
  await waitFor(() => expect(client.getQueryState(["projects", "detail", project.id])?.status).toBe("success"));
  vi.mocked(api.project).mockRejectedValue(new Error("Selected project refresh failed"));
  await client.invalidateQueries({ queryKey: ["projects", "detail", project.id] });
  await screen.findByText("Selected project refresh failed");
  expect(screen.queryByRole("option", { name: "Saved notebook" })).toBeNull();
  expect(screen.getByRole("option", { name: "Selected project" })).toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id);
  expect(change).not.toHaveBeenCalled();
});
