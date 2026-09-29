import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ReferenceDetail } from "./ReferenceDetail";
import type { ReferenceSubject } from "./types";

vi.mock("./api", () => ({
  api: { referenceAssets: vi.fn(), updateReference: vi.fn() },
}));

const subject: ReferenceSubject = {
  id: "reference", name: "Keyboard reference", mention_slug: "keyboard-reference",
  kind: "object", description: null, aliases_json: [], tags_json: [],
  cover_artifact_id: null, favorite: false, archived: false,
};

function show() {
  vi.mocked(api.referenceAssets).mockResolvedValue([]);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <ReferenceDetail subject={subject} onBack={() => {}} />
    </QueryClientProvider>,
  );
}

afterEach(cleanup);

it("retains the Save action through a pending save and its canonical response", async () => {
  let finish!: (value: ReferenceSubject) => void;
  vi.mocked(api.updateReference).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  show();
  fireEvent.change(screen.getByLabelText("Other names, separated by commas"), {
    target: { value: "  Blue block, blue block " },
  });
  const save = screen.getByRole("button", { name: "Save details" });
  save.focus();
  fireEvent.click(save);
  await waitFor(() => expect(api.updateReference).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(save.getAttribute("aria-disabled")).toBe("true"));
  expect(save.hasAttribute("disabled")).toBe(false);
  expect(document.activeElement).toBe(save);
  await act(async () => { fireEvent.click(save); });
  expect(api.updateReference).toHaveBeenCalledTimes(1);
  await act(async () => { finish({ ...subject, aliases_json: ["Blue block"] }); });
  await waitFor(() => expect(screen.getByLabelText("Other names, separated by commas")).toHaveProperty("value", "Blue block"));
  expect(document.activeElement).toBe(save);
  expect(save.getAttribute("aria-disabled")).toBe("true");
  await act(async () => { fireEvent.click(save); });
  expect(api.updateReference).toHaveBeenCalledTimes(1);
});

it("keeps a failed save focused and permits a deliberate retry", async () => {
  let refuse!: (reason: Error) => void;
  vi.mocked(api.updateReference).mockImplementationOnce(() => new Promise((_resolve, reject) => { refuse = reject; }));
  vi.mocked(api.updateReference).mockResolvedValueOnce({ ...subject, description: "Blue block" });
  show();
  fireEvent.change(screen.getByLabelText("Description"), { target: { value: "Blue block" } });
  const save = screen.getByRole("button", { name: "Save details" });
  save.focus();
  fireEvent.click(save);
  await waitFor(() => expect(api.updateReference).toHaveBeenCalledTimes(1));
  expect(save.hasAttribute("disabled")).toBe(false);
  await act(async () => { refuse(new Error("Save unavailable")); });
  expect(await screen.findByText("Save unavailable")).toBeTruthy();
  expect(document.activeElement).toBe(save);
  expect(save.getAttribute("aria-disabled")).toBe("false");
  fireEvent.click(save);
  await waitFor(() => expect(api.updateReference).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(save.getAttribute("aria-disabled")).toBe("true"));
});

it("ignores Save when no details have changed", async () => {
  show();
  const save = screen.getByRole("button", { name: "Save details" });
  expect(save.getAttribute("aria-disabled")).toBe("true");
  expect(save.hasAttribute("disabled")).toBe(false);
  save.focus();
  await act(async () => { fireEvent.click(save); });
  expect(api.updateReference).not.toHaveBeenCalled();
  expect(document.activeElement).toBe(save);
});
