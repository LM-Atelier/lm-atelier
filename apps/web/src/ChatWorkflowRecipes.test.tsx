import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ChatWorkflowChoices } from "./ChatWorkflowChoices";

vi.mock("./ActiveChatWorkflowSelector", () => ({
  ActiveChatWorkflowSelector: ({ label }: { label: string }) => <span>{label}</span>,
}));
afterEach(() => { cleanup(); localStorage.clear(); });

it("keeps recipes out of the composer while offering each workflow type", () => {
  render(<ChatWorkflowChoices chatId="new-chat" routingMode="auto" />);
  expect(screen.queryByText("Recipes for this chat")).not.toBeInTheDocument();
  for (const label of ["Text workflow", "Image workflow", "Video workflow"]) expect(screen.getByText(label)).toBeVisible();
});

it("remembers hidden workflow controls for this chat without hiding another chat's controls", () => {
  const { rerender } = render(<ChatWorkflowChoices chatId="first-chat" routingMode="auto" />);
  fireEvent.click(screen.getByRole("button", { name: "Hide workflows" }));
  expect(screen.queryByText("Text workflow")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Show workflows" })).toHaveAttribute("aria-expanded", "false");
  rerender(<ChatWorkflowChoices chatId="second-chat" routingMode="image" />);
  expect(screen.getByText("Text workflow")).toBeVisible();
  rerender(<ChatWorkflowChoices chatId="first-chat" routingMode="video" />);
  expect(screen.queryByText("Text workflow")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Show workflows" }));
  expect(screen.getByText("Text workflow")).toBeVisible();
});

it("can hide and show controls when browser storage is unavailable", () => {
  const read = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("Unavailable"); });
  const write = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("Unavailable"); });
  try {
    render(<ChatWorkflowChoices chatId="new-chat" routingMode="auto" />);
    fireEvent.click(screen.getByRole("button", { name: "Hide workflows" }));
    expect(screen.queryByText("Image workflow")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show workflows" }));
    expect(screen.getByText("Image workflow")).toBeVisible();
  } finally { read.mockRestore(); write.mockRestore(); }
});
