import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ModelUpdateProfileOffers } from "./ModelUpdateProfileOffers";
import type { Job, ModelInstall, ModelProfile } from "./types";

const pagedProfiles = vi.hoisted(() => vi.fn());
vi.mock("./api", () => ({ api: {
  downloadJob: vi.fn(), modelInstall: vi.fn(), profiles: vi.fn(), profilesPage: pagedProfiles,
  updateProfileModel: vi.fn(),
} }));
const profile = {
  id: "profile-first", name: "First profile", model_install_id: "previous-install",
  role: "image", engine: "comfyui",
} as ModelProfile;

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <ModelUpdateProfileOffers
      downloads={[{ jobId: "update-job", previousInstallId: "previous-install", modelName: "Landscape" }]}
      onDismiss={vi.fn()}
    />
  </QueryClientProvider>);
}

beforeEach(() => {
  vi.mocked(api.downloadJob).mockResolvedValue({ status: "complete", result_json: { model_install_id: "replacement" } } as unknown as Job);
  vi.mocked(api.modelInstall).mockResolvedValue({ id: "replacement", active: true, readiness: "ready", role: "image", engine: "comfyui" } as ModelInstall);
  vi.mocked(api.profiles).mockResolvedValue([profile]);
  pagedProfiles.mockResolvedValue([profile]);
});
afterEach(cleanup);

it("asks for a bounded page matching the previous install and replacement engine", async () => {
  show();
  await screen.findByRole("button", { name: "Switch First profile" });
  expect(pagedProfiles).toHaveBeenCalledWith({
    limit: 50, offset: 0, installIds: ["previous-install"], role: "image", engine: "comfyui",
  });
  expect(api.profiles).not.toHaveBeenCalled();
});

it("loads matching profiles beyond the first page without losing earlier choices", async () => {
  const firstPage = Array.from({ length: 50 }, (_, index) => ({ ...profile, id: `profile-${index}`, name: `Profile ${index}` }));
  pagedProfiles.mockImplementation(({ offset }: { offset: number }) => Promise.resolve(
    offset === 0 ? firstPage : [{ ...profile, id: "profile-later", name: "Later profile" }],
  ));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "More profiles" }));
  await screen.findByRole("button", { name: "Switch Later profile" });
  expect(screen.getByRole("button", { name: "Switch Profile 0" })).toBeInTheDocument();
  expect(pagedProfiles).toHaveBeenLastCalledWith({
    limit: 50, offset: 50, installIds: ["previous-install"], role: "image", engine: "comfyui",
  });
  expect(screen.queryByRole("button", { name: "More profiles" })).not.toBeInTheDocument();
});

it("offers retry without claiming no profiles exist after a failed read", async () => {
  vi.mocked(api.profiles).mockRejectedValue(new Error("Profiles temporarily unavailable"));
  pagedProfiles.mockRejectedValueOnce(new Error("Profiles temporarily unavailable")).mockResolvedValue([profile]);
  show();
  await screen.findByText("Profiles temporarily unavailable");
  expect(screen.queryByText("No profiles use the previous version.")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry profiles" }));
  await screen.findByRole("button", { name: "Switch First profile" });
});
