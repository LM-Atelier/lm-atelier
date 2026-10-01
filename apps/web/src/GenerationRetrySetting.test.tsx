import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { GenerationRetrySetting } from "./GenerationRetrySetting";
import { api } from "./api";

vi.mock("./api", () => ({ api: { generationRetryPolicy: vi.fn(), updateGenerationRetryPolicy: vi.fn() } }));
afterEach(() => { cleanup(); vi.resetAllMocks(); });

describe("automatic media retry setting", () => {
  it("shows the persisted default of three with cancellation and zero explained", async () => {
    vi.mocked(api.generationRetryPolicy).mockResolvedValue({ max_retries: 3, revision: 0 });
    render(<GenerationRetrySetting />);
    await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Automatic retries" })).toHaveValue(3));
    expect(screen.getByText(/Images and videos only/)).toHaveTextContent("Cancelled work never retries. Set 0 to turn this off.");
    expect(screen.getByRole("button", { name: "Save retries" })).toHaveAttribute("aria-disabled", "true");
  });

  it("saves zero with the observed revision and preserves keyboard focus", async () => {
    vi.mocked(api.generationRetryPolicy).mockResolvedValue({ max_retries: 3, revision: 5 });
    vi.mocked(api.updateGenerationRetryPolicy).mockResolvedValue({ max_retries: 0, revision: 6 });
    render(<GenerationRetrySetting />);
    const input = await screen.findByRole("spinbutton", { name: "Automatic retries" });
    await waitFor(() => expect(input).toHaveValue(3));
    fireEvent.change(input, { target: { value: "0" } });
    const save = screen.getByRole("button", { name: "Save retries" });
    save.focus();
    fireEvent.click(save);
    expect(await screen.findByText("Automatic retry setting saved.")).toBeVisible();
    expect(api.updateGenerationRetryPolicy).toHaveBeenCalledExactlyOnceWith(0, 5);
    expect(input).toHaveValue(0);
    expect(save).toHaveFocus();
  });

  it.each(["", "-1", "1.5", "11"])("refuses invalid retry count %s without sending a write", async (value) => {
    vi.mocked(api.generationRetryPolicy).mockResolvedValue({ max_retries: 3, revision: 0 });
    render(<GenerationRetrySetting />);
    const input = await screen.findByRole("spinbutton", { name: "Automatic retries" });
    await waitFor(() => expect(input).toHaveValue(3));
    fireEvent.change(input, { target: { value } });
    const save = screen.getByRole("button", { name: "Save retries" });
    expect(save).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(save);
    expect(api.updateGenerationRetryPolicy).not.toHaveBeenCalled();
  });

  it("keeps a failed save local and requires a refresh before using a new revision", async () => {
    vi.mocked(api.generationRetryPolicy)
      .mockResolvedValueOnce({ max_retries: 3, revision: 2 })
      .mockResolvedValueOnce({ max_retries: 7, revision: 3 });
    vi.mocked(api.updateGenerationRetryPolicy).mockRejectedValue(new Error("neutral internal error marker"));
    render(<GenerationRetrySetting />);
    const input = await screen.findByRole("spinbutton", { name: "Automatic retries" });
    await waitFor(() => expect(input).toHaveValue(3));
    fireEvent.change(input, { target: { value: "5" } });
    fireEvent.click(screen.getByRole("button", { name: "Save retries" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The retry setting could not be saved. Refresh and try again.");
    expect(screen.queryByText("neutral internal error marker")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Save retries" }));
    expect(api.updateGenerationRetryPolicy).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Refresh retry settings" }));
    await waitFor(() => expect(input).toHaveValue(7));
  });

  it("cannot turn a load failure into a default-backed save", async () => {
    vi.mocked(api.generationRetryPolicy).mockRejectedValue(new Error("neutral read failure"));
    render(<GenerationRetrySetting />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Generation retry settings could not be loaded.");
    fireEvent.click(screen.getByRole("button", { name: "Save retries" }));
    expect(api.updateGenerationRetryPolicy).not.toHaveBeenCalled();
  });
});
