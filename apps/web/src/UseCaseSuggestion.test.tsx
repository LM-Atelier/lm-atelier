import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import type { ModelAssetInstall, ModelProfile, UseCaseSuggestionOut } from "./types";
import { UseCaseSuggestion } from "./UseCaseSuggestion";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ...actual, api: { ...actual.api,
    suggestProfileUseCase: vi.fn(), suggestLoraUseCase: vi.fn(),
    updateProfile: vi.fn(), updateModelAsset: vi.fn(),
  } };
});

const updated = { id: "target", use_case: "Watercolor scenes.", use_case_derived: true };
beforeEach(() => {
  vi.mocked(api.suggestProfileUseCase).mockReset().mockResolvedValue({ suggestion: "Watercolor scenes." });
  vi.mocked(api.suggestLoraUseCase).mockReset().mockResolvedValue({ suggestion: "Watercolor scenes." });
  vi.mocked(api.updateProfile).mockReset().mockResolvedValue(updated as ModelProfile);
  vi.mocked(api.updateModelAsset).mockReset().mockResolvedValue(updated as ModelAssetInstall);
});
afterEach(cleanup);

function show(kind: "profile" | "lora" = "profile", available = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const view = (savedText: string) => <QueryClientProvider client={client}>
    <UseCaseSuggestion kind={kind} id="target" name="Watercolor" savedText={savedText} available={available} />
  </QueryClientProvider>;
  const result = render(view("My existing text"));
  return { client, update: (value: string) => result.rerender(view(value)) };
}

function open() {
  const button = screen.getByRole("button", { name: "Suggest use case for Watercolor" });
  button.focus();
  fireEvent.click(button);
  return button;
}

it.each(["profile", "lora"] as const)("keeps a %s suggestion unsaved until explicit confirmation", async (kind) => {
  const { client } = show(kind);
  const opener = open();
  expect(await screen.findByRole("textbox", { name: "Suggested use case for Watercolor" })).toHaveValue("Watercolor scenes.");
  expect(screen.getByText("Current use case: My existing text")).toBeVisible();
  expect(api.updateProfile).not.toHaveBeenCalled();
  expect(api.updateModelAsset).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Save use case" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  const save = kind === "profile" ? api.updateProfile : api.updateModelAsset;
  expect(save).toHaveBeenCalledExactlyOnceWith("target", {
    use_case: "Watercolor scenes.", use_case_derived: true, expected_use_case: "My existing text",
  });
  expect(client.getQueryData([kind === "profile" ? "profiles" : "model-assets"])).toEqual([updated]);
  // The save closes the dialog once its request settles, and the dialog hands
  // focus back as it is taken down, a moment after it has left the page.
  await waitFor(() => expect(opener).toHaveFocus());
});

it("labels an edited suggestion as manual when saving", async () => {
  show();
  open();
  const field = await screen.findByRole("textbox", { name: "Suggested use case for Watercolor" });
  fireEvent.change(field, { target: { value: "My watercolor and ink notes" } });
  expect(screen.getByText("Your edited text will be saved as a manual use case.")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Save use case" }));
  await waitFor(() => expect(api.updateProfile).toHaveBeenCalledWith("target", {
    use_case: "My watercolor and ink notes", use_case_derived: false, expected_use_case: "My existing text",
  }));
});

it("Escape cancels pending generation and ignores a late result", async () => {
  let finish!: (value: UseCaseSuggestionOut) => void;
  vi.mocked(api.suggestProfileUseCase).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  show();
  const opener = open();
  expect(screen.getByRole("status")).toHaveTextContent("Generating");
  const signal = vi.mocked(api.suggestProfileUseCase).mock.calls[0][2];
  fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
  expect(signal?.aborted).toBe(true);
  expect(opener).toHaveFocus();
  await act(async () => { finish({ suggestion: "Late suggestion" }); });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(api.updateProfile).not.toHaveBeenCalled();
});

it("cancelling a completed suggestion preserves saved text", async () => {
  show("lora");
  const opener = open();
  await screen.findByRole("textbox");
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(api.updateModelAsset).not.toHaveBeenCalled();
  expect(opener).toHaveFocus();
});

it("refuses to save after the current use case changes", async () => {
  const { update } = show();
  open();
  await screen.findByRole("textbox");
  update("A newer manual edit");
  expect(screen.getByRole("alert")).toHaveTextContent("saved use case changed");
  const save = screen.getByRole("button", { name: "Save use case" });
  expect(save).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(save);
  expect(api.updateProfile).not.toHaveBeenCalled();
  expect(api.suggestProfileUseCase).toHaveBeenCalledTimes(1);
});

it("keeps focus and prevents duplicate saves while a save is pending", async () => {
  let finish!: (value: ModelProfile) => void;
  vi.mocked(api.updateProfile).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  show();
  open();
  await screen.findByRole("textbox");
  const save = screen.getByRole("button", { name: "Save use case" });
  save.focus();
  fireEvent.click(save);
  expect(save).toHaveFocus();
  expect(save).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(save);
  fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
  expect(screen.getByRole("dialog")).toBeVisible();
  expect(api.updateProfile).toHaveBeenCalledTimes(1);
  await act(async () => { finish(updated as ModelProfile); });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("keeps the suggestion editable when saving fails", async () => {
  vi.mocked(api.updateProfile).mockRejectedValue(new Error("The saved use case changed."));
  show();
  open();
  await screen.findByRole("textbox");
  fireEvent.click(screen.getByRole("button", { name: "Save use case" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("saved use case changed");
  expect(screen.getByRole("textbox")).toHaveValue("Watercolor scenes.");
  expect(screen.getByRole("button", { name: "Save use case" })).toHaveAttribute("aria-disabled", "false");
});

it("explains a missing local worker without changing saved text", async () => {
  vi.mocked(api.suggestProfileUseCase).mockRejectedValue(new Error("Start a local chat model."));
  show();
  open();
  expect(await screen.findByRole("alert")).toHaveTextContent("Start a local chat model.");
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  expect(api.updateProfile).not.toHaveBeenCalled();
});

it("does not offer inference when no saved description is available", () => {
  show("profile", false);
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
  expect(api.suggestProfileUseCase).not.toHaveBeenCalled();
});
