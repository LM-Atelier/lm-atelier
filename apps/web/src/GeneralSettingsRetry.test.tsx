import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { GeneralSettings } from "./GeneralSettings";

vi.mock("./api", () => ({ api: {
  generationRetryPolicy: vi.fn(async () => ({ max_retries: 3, revision: 0 })),
  updateGenerationRetryPolicy: vi.fn(),
} }));
afterEach(cleanup);

it("offers the saved media retry allowance in General settings", async () => {
  render(<GeneralSettings />);
  expect(await screen.findByRole("spinbutton", { name: "Automatic retries" })).toHaveValue(3);
});
