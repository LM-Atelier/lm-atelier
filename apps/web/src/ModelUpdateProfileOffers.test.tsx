import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ModelUpdateProfileOffers } from "./ModelUpdateProfileOffers";
import type { Job, ModelInstall, ModelProfile } from "./types";

vi.mock("./api", () => ({ api: {
  downloadJob: vi.fn(), modelInstall: vi.fn(), profiles: vi.fn(), updateProfileModel: vi.fn(),
} }));

const download = { jobId: "download-a", previousInstallId: "old-install", modelName: "Landscape" };
const profile = {
  id: "profile-a", name: "My landscapes", model_install_id: "old-install", role: "image", engine: "comfyui",
} as ModelProfile;

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const onDismiss = vi.fn();
  render(<QueryClientProvider client={client}>
    <ModelUpdateProfileOffers downloads={[download]} onDismiss={onDismiss} />
  </QueryClientProvider>);
  return { client, onDismiss };
}

beforeEach(() => {
  vi.mocked(api.downloadJob).mockResolvedValue({ status: "complete", result_json: { model_install_id: "new-install" } } as unknown as Job);
  vi.mocked(api.modelInstall).mockResolvedValue({ id: "new-install", active: true, readiness: "ready", role: "image", engine: "comfyui" } as ModelInstall);
  vi.mocked(api.profiles).mockResolvedValue([profile]);
  vi.mocked(api.updateProfileModel).mockResolvedValue({ ...profile, model_install_id: "new-install" });
});
afterEach(cleanup);

it("offers an exact completed update and switches only the chosen profile on a click", async () => {
  vi.mocked(api.profiles).mockResolvedValue([
    profile, { ...profile, id: "other", name: "Other", model_install_id: "unrelated" },
    { ...profile, id: "wrong-role", name: "Wrong role", role: "chat" },
  ]);
  show();
  const button = await screen.findByRole("button", { name: "Switch My landscapes" });
  expect(api.downloadJob).toHaveBeenCalledWith("download-a");
  expect(api.modelInstall).toHaveBeenCalledWith("new-install");
  expect(api.updateProfileModel).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", { name: "Switch Other" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Switch Wrong role" })).not.toBeInTheDocument();
  vi.mocked(api.profiles).mockResolvedValue([{ ...profile, model_install_id: "new-install" }]);
  fireEvent.click(button);
  await screen.findByText("Updated My landscapes.");
  expect(api.updateProfileModel).toHaveBeenCalledExactlyOnceWith("profile-a", {
    expected_install_id: "old-install", download_job_id: "download-a",
  });
  expect(screen.queryByRole("button", { name: "Switch My landscapes" })).not.toBeInTheDocument();
});

it.each(["queued", "running", "paused", "failed", "cancelled"])("never offers a %s download", async (status) => {
  vi.mocked(api.downloadJob).mockResolvedValue({ status, result_json: { model_install_id: "new-install" } } as unknown as Job);
  const { client } = show();
  await waitFor(() => expect(client.getQueryState(["model-update-download", "download-a"])?.status).toBe("success"));
  expect(api.modelInstall).not.toHaveBeenCalled();
  expect(api.updateProfileModel).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", { name: /Switch/ })).not.toBeInTheDocument();
});

it("shows the offer after the tracked download completes", async () => {
  vi.mocked(api.downloadJob).mockResolvedValue({ status: "running", result_json: {} } as unknown as Job);
  const { client } = show();
  await waitFor(() => expect(client.getQueryState(["model-update-download", "download-a"])?.status).toBe("success"));
  vi.mocked(api.downloadJob).mockResolvedValue({ status: "complete", result_json: { model_install_id: "new-install" } } as unknown as Job);
  await client.invalidateQueries({ queryKey: ["model-update-download", "download-a"] });
  await screen.findByRole("button", { name: "Switch My landscapes" });
  expect(api.updateProfileModel).not.toHaveBeenCalled();
});

it.each([
  { active: false, readiness: "ready" }, { active: true, readiness: "unverified" },
])("does not offer an inactive or unverified installation: %j", async (state) => {
  vi.mocked(api.modelInstall).mockResolvedValue({ ...state, id: "new-install", role: "image", engine: "comfyui" } as ModelInstall);
  show();
  await screen.findByText(/needs current runtime verification/);
  expect(api.profiles).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", { name: /Switch/ })).not.toBeInTheDocument();
});

it("retains a refused switch and presents its error", async () => {
  vi.mocked(api.updateProfileModel).mockRejectedValue(new Error("This profile changed. Refresh it before switching models."));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Switch My landscapes" }));
  await screen.findByText("This profile changed. Refresh it before switching models.");
  expect(screen.queryByText("Updated My landscapes.")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Switch My landscapes" })).toBeInTheDocument();
});

it("dismisses without changing profiles", async () => {
  const { onDismiss } = show();
  fireEvent.click(await screen.findByRole("button", { name: "Dismiss update" }));
  expect(onDismiss).toHaveBeenCalledExactlyOnceWith("download-a");
  expect(api.updateProfileModel).not.toHaveBeenCalled();
});
