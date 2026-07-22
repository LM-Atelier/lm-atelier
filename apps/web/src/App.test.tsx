import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./api", () => ({
  api: {
    initialize: vi.fn().mockResolvedValue(undefined),
    projects: vi.fn().mockResolvedValue([]),
    chats: vi.fn().mockResolvedValue([]),
    chat: vi.fn(),
    createProject: vi.fn(),
    createChat: vi.fn(),
    sendTurn: vi.fn(),
    jobs: vi.fn().mockResolvedValue([]),
    cancelJob: vi.fn(),
    engines: vi.fn().mockResolvedValue([
      {
        engine: "mock",
        version: "1",
        roles: ["chat", "image", "video"],
        operations: ["text", "text_to_image", "text_to_video"],
        formats: ["mock"],
        devices: ["cpu:0"],
        streaming: true,
        tool_calling: true,
        settings: [],
        healthy: true,
        details: {},
      },
    ]),
    system: vi.fn(),
    models: vi.fn(),
    profiles: vi.fn().mockResolvedValue([]),
    catalog: vi.fn(),
    recipes: vi.fn().mockResolvedValue([]),
    installRecipe: vi.fn(),
    download: vi.fn(),
    workflows: vi.fn(),
    createWorkflow: vi.fn(),
    validateWorkflow: vi.fn(),
    upload: vi.fn(),
  },
  connectEvents: vi.fn().mockImplementation(async (_onEvent, onStatus) => {
    onStatus(true);
    return () => undefined;
  }),
}));

describe("App", () => {
  beforeEach(() => localStorage.clear());

  it("renders the local workspace shell without an existing chat", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("Start a local conversation")).toBeInTheDocument();
    expect(screen.getByText("Model library")).toBeInTheDocument();
    expect(screen.getByText("Local service connected")).toBeInTheDocument();
  });
});
