import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { expect, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import type { EngineCapabilities, Workflow } from "./types";

export async function exercisePriorImageWorkflowControls(
  roleAwareMediaEngine: EngineCapabilities,
  setWorkflowFixtures: (values: Workflow[]) => void,
) {
  const stamp = "2026-07-22T00:00:00Z";
  const chat = {
    id: "chat-prior-image-controls",
    project_id: null,
    title: "Prior image controls",
    pinned: false, archived: false,
    routing_mode: "image" as const,
    confirm_uncertain_media: false,
    active_chat_profile_id: null,
    active_image_profile_id: null,
    active_video_profile_id: null,
    active_head_message_id: "assistant-prior-image",
    created_at: stamp,
    updated_at: stamp,
  };
  const userMessage = {
    id: "user-prior-image",
    chat_id: chat.id,
    parent_id: null,
    role: "user" as const,
    status: "complete" as const,
    parts: [{ id: "prior-prompt", position: 0, type: "text" as const, text: "Make an apple", artifact_id: null, metadata_json: {} }],
    created_at: stamp,
    updated_at: stamp,
  };
  const assistantMessage = {
    id: "assistant-prior-image",
    chat_id: chat.id,
    parent_id: userMessage.id,
    role: "assistant" as const,
    status: "complete" as const,
    parts: [{ id: "prior-image", position: 0, type: "image" as const, text: null, artifact_id: "sha256:prior", metadata_json: {} }],
    created_at: stamp,
    updated_at: stamp,
  };
  const workflow = (
    id: string,
    operation: "text_to_image" | "image_to_image",
    title: string,
  ) => ({
    id,
    name: title,
    operation,
    description: "",
    current_revision_id: `${id}-revision`,
    revisions: [{
      id: `${id}-revision`,
      workflow_id: id,
      version: 1,
      engine: "mock",
      engine_version: null,
      ui_graph_json: {},
      api_graph_json: {},
      input_schema_json: {
        type: "object",
        properties: {
          negative_prompt: {
            type: "string",
            title,
            default: "",
          },
          ...(operation === "image_to_image" ? {
            strength: {
              type: "number",
              title: "Edit strength",
              default: 0.9,
              minimum: 0,
              maximum: 1,
              "x-lm-atelier-visibility": "basic",
            },
            steps: { type: "integer", title: "Steps", default: 4 },
          } : {}),
        },
        ...(operation === "image_to_image" ? {
          "x-lm-atelier-edit-calibration": {
            version: 1,
            edit_strength: {
              parameter: "strength",
              minimum: 0,
              maximum: 1,
              recommended: {
                minimal: 0.3,
                localized: 0.45,
                replacement: 0.6,
                global: 0.8,
                fallback: 0.5,
              },
            },
            schedule: {
              steps_parameter: "steps",
              minimum_effective_steps: {
                localized: 2,
                replacement: 3,
                global: 3,
              },
            },
          },
        } : {}),
      },
      dependencies_json: {},
      trusted: true,
      created_at: stamp,
    }],
  });
  localStorage.setItem("local-lm-chat", chat.id);
  vi.mocked(api.engines).mockResolvedValue([roleAwareMediaEngine]);
  vi.mocked(api.chats).mockResolvedValue([chat]);
  vi.mocked(api.chat).mockResolvedValue({
    ...chat,
    messages: [userMessage, assistantMessage],
  });
  setWorkflowFixtures([
    workflow("fresh-image", "text_to_image", "Fresh image exclusion"),
    workflow("edit-image", "image_to_image", "Edit image exclusion"),
  ]);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );

  // The router, not the browser, decides whether a draft reuses the prior
  // image; this stands in for it with the answers the server gives.
  vi.mocked(api.classifyDraft).mockImplementation(async (_chatId, draft) => ({
    references_prior_visual: draft.trim() === "Make it green",
  }));

  const composer = await screen.findByRole("textbox", { name: "Message" });
  fireEvent.change(composer, { target: { value: "Create an image of a pear" } });
  await waitFor(() => expect(api.classifyDraft).toHaveBeenCalledWith(
    chat.id,
    "Create an image of a pear",
    "image",
  ));
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  expect(screen.getByLabelText(/Fresh image exclusion/)).toBeInTheDocument();
  expect(screen.queryByLabelText(/Edit image exclusion/)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Close settings" }));

  fireEvent.change(composer, { target: { value: "Make it green" } });
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  await waitFor(() => expect(screen.getByLabelText(/Edit image exclusion/)).toBeInTheDocument());
  expect(screen.queryByLabelText(/Fresh image exclusion/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText("Predicted: localized")).toBeInTheDocument();
  expect(screen.queryByLabelText("Manual change strength")).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Manual" }));
  expect(screen.getByLabelText("Manual change strength")).toHaveValue(0.5);
  await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(chat.id, {
    generation_settings_json: {
      image: {
        _image_edit_strength_mode: "manual",
        strength: 0.5,
      },
    },
  }));
  fireEvent.click(screen.getByRole("button", { name: "Auto" }));
  expect(screen.queryByLabelText("Manual change strength")).not.toBeInTheDocument();
  await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(chat.id, {
    generation_settings_json: {
      image: { _image_edit_strength_mode: "auto" },
    },
  }));
  fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
  fireEvent.click(screen.getByRole("button", { name: "Edit this image" }));
  await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(chat.id, { routing_mode: "image" }));
  expect(screen.getByText("sha256:prior")).toBeInTheDocument();
  expect(composer).toHaveFocus();
}
