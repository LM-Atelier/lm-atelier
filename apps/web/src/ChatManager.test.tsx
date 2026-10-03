import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ChatManager } from "./ChatManager";
import type { Chat } from "./types";

vi.mock("./ProjectPicker", () => ({ ProjectPicker: ({ value, onChange }: { value: string; onChange: (value: string) => void }) =>
  <label>Project<select value={value} onChange={(event) => onChange(event.target.value)}><option value="">Unfiled</option><option value="project-a">Garden</option><option value="project-b">Sketches</option></select></label>,
}));
vi.mock("./ChatWebAccess", () => ({ ChatWebAccess: () => null }));
vi.mock("./WorkflowRecipeChoices", () => ({ ChatWorkflowRecipes: () => null }));
vi.mock("./ChatTrashConfirmation", () => ({ ChatTrashConfirmation: () => null }));
afterEach(cleanup);

function show(projectId: string | null) {
  const save = vi.fn();
  const chat = { id: "chat-a", title: "Garden notes", project_id: projectId, archived: false, confirm_uncertain_media: true } as Chat;
  render(<ChatManager chat={chat} onSave={save} onClose={vi.fn()} onDelete={vi.fn()} />);
  return save;
}

it.each([null, "project-a"])("saves a renamed chat without writing its unchanged filing (%s)", (projectId) => {
  const save = show(projectId);
  fireEvent.change(screen.getByLabelText("Title"), { target: { value: "New garden notes" } });
  fireEvent.click(screen.getByRole("button", { name: "Save chat" }));
  expect(save).toHaveBeenCalledWith(expect.objectContaining({ title: "New garden notes" }));
  expect(save.mock.calls[0][0]).not.toHaveProperty("project_id");
});

it.each(["project-b", ""])("saves a deliberate filing change (%s)", (projectId) => {
  const save = show("project-a");
  fireEvent.change(screen.getByLabelText("Project"), { target: { value: projectId } });
  fireEvent.click(screen.getByRole("button", { name: "Save chat" }));
  expect(save).toHaveBeenCalledWith(expect.objectContaining({ project_id: projectId || null }));
});
