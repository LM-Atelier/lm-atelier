/** The runtime card says which runtime is in use when another version installed it. */

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { RuntimeSetupCard } from "./RuntimeSetupCard";
import type { RuntimeStatus } from "./types";

afterEach(cleanup);

const missing: RuntimeStatus = {
  engine: "comfyui",
  release: "v0.28.0",
  state: "missing",
  supported: true,
  managed: false,
  progress: 0,
  downloaded_bytes: 0,
  size_bytes: 2_092_156_323,
  distribution: "external-gpl-3.0",
  license: "GPL-3.0-only",
  message: "Installs automatically when first used.",
};

it("names another version's runtime that is still in use and still offers the install", () => {
  const message =
    "ComfyUI v0.30.0, installed by another version of LM Atelier, is still in use."
    + " This version uses v0.28.0. Install it to switch.";
  render(
    <RuntimeSetupCard
      runtime={{ ...missing, installed_release: "v0.30.0", message }}
      installPending={false}
      onInstall={() => undefined}
    />,
  );

  expect(screen.getByText(message)).toBeInTheDocument();
  expect(screen.queryByText(/downloaded separately/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^Install/ })).toBeInTheDocument();
});

it("keeps the licence line for a runtime that is simply not installed", () => {
  render(<RuntimeSetupCard runtime={missing} installPending={false} onInstall={() => undefined} />);

  expect(screen.getByText("GPL-3.0-only · downloaded separately")).toBeInTheDocument();
  expect(screen.queryByText(missing.message)).not.toBeInTheDocument();
});
