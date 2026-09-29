import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ChatWorkflowChoices } from "./ChatWorkflowChoices";

vi.mock("./ActiveChatWorkflowSelector", () => ({
  ActiveChatWorkflowSelector: ({ label }: { label: string }) => <span>{label}</span>,
}));
vi.mock("./api", () => ({ api: {
  workflowUseCasePresets: vi.fn(async () => []),
  workflowUseCaseChoice: vi.fn(async () => ({ mode: "inherit" })),
} }));
afterEach(cleanup);

it("offers recipe choices in Auto while leaving every workflow type visible", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(<QueryClientProvider client={client}><ChatWorkflowChoices chatId="new-chat" routingMode="auto" /></QueryClientProvider>);
  const summary = screen.getByText("Recipes for this chat");
  expect(api.workflowUseCasePresets).not.toHaveBeenCalled();
  expect(screen.getByText("Text workflow")).toBeVisible();
  expect(screen.getByText("Image workflow")).toBeVisible();
  expect(screen.getByText("Video workflow")).toBeVisible();
  fireEvent.click(summary);
  fireEvent(summary.parentElement!, new Event("toggle"));
  await waitFor(() => expect(screen.getAllByRole("combobox")).toHaveLength(8));
  expect(screen.getByText("Auto chooses the request type at send.")).toBeVisible();
  client.clear();
});
