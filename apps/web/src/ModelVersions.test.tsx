import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { ModelCard } from "./ModelCard";
import type { CatalogModel, CatalogVersionRow } from "./types";
import { VersionChooser } from "./VersionChooser";

const imageRole = "image";
const model: CatalogModel = {
  provider: "civitai",
  remote_id: "102",
  parent_model_id: "100",
  parent_model_name: "Landscape model",
  name: "Landscape model",
  author: "Example",
  pipeline_tag: "text-to-image",
  tags: [],
  downloads: 0,
  likes: 0,
  trending_score: null,
  created_at: null,
  last_modified: null,
  gated: false,
  private: false,
  library_name: null,
  architecture: null,
  formats: ["safetensors"],
  quantizations: [],
  parameter_count: null,
  license_id: null,
  total_size_bytes: null,
  compatibility: "likely",
  compatibility_reasons: [],
  version_count: 2,
  installed_version_count: 1,
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("grouped model versions", () => {
  it.each([1, 2])("keeps the chooser available with %i installed versions", (count) => {
    const choose = vi.fn();
    const download = vi.fn();
    render(
      <ModelCard model={{ ...model, installed_version_count: count }} role={imageRole}
        status="installed" onDownload={download} onChooseVersion={choose} />,
    );

    expect(screen.getByText(`${count} of 2 installed`)).toBeInTheDocument();
    const manage = screen.getByRole("button", { name: "Manage 2 versions" });
    expect(manage).toBeEnabled();
    fireEvent.click(manage);
    expect(choose).toHaveBeenCalledTimes(1);
    expect(download).not.toHaveBeenCalled();
  });

  it.each([null, 0])("does not invent an installed count from %s", (count) => {
    render(
      <ModelCard model={{ ...model, installed_version_count: count }} role={imageRole}
        status="idle" onDownload={vi.fn()} onChooseVersion={vi.fn()} />,
    );
    expect(screen.queryByText(/of 2 installed/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose from 2 versions" })).toBeEnabled();
  });

  it.each(["preparing", "downloading"] as const)("keeps a %s card disabled", (status) => {
    const choose = vi.fn();
    render(<ModelCard model={model} role={imageRole} status={status}
      onDownload={vi.fn()} onChooseVersion={choose} />);
    const action = screen.getByRole("button");
    expect(action).toBeDisabled();
    fireEvent.click(action);
    expect(choose).not.toHaveBeenCalled();
  });

  it("does not reinstall an installed single version", () => {
    render(<ModelCard model={{ ...model, version_count: 1 }} role={imageRole} status="installed"
      onDownload={vi.fn()} onChooseVersion={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Installed" })).toBeDisabled();
  });

  it("keeps unsupported grouped models disabled", () => {
    render(<ModelCard model={{ ...model, compatibility: "unsupported" }} role={imageRole}
      status="installed" onDownload={vi.fn()} onChooseVersion={vi.fn()} />);
    expect(screen.getByRole("button")).toBeDisabled();
  });
});

function renderChooser(versions: CatalogVersionRow[]) {
  vi.spyOn(api, "catalogVersions").mockResolvedValue({ model_id: "100", versions });
  const choose = vi.fn();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <VersionChooser modelId="100" modelName="Landscape model" onChoose={choose} onClose={vi.fn()} />
  </QueryClientProvider>);
  return choose;
}

describe("version chooser installed state", () => {
  it("shows every installed version without an installable action", async () => {
    renderChooser([
      { version_id: "102", version_name: "Second edition", size_bytes: 10, installed: true },
      { version_id: "101", version_name: "First edition", size_bytes: 10, installed: true },
    ]);
    const installed = await screen.findAllByRole("button", { name: "Installed" });
    expect(installed).toHaveLength(2);
    for (const button of installed) expect(button).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Install this version" })).not.toBeInTheDocument();
    expect(api.catalogVersions).toHaveBeenCalledWith("100");
  });

  it("selects the exact available version and preserves unknown state", async () => {
    const choose = renderChooser([
      { version_id: "103", version_name: "Third edition", size_bytes: 10, installed: true },
      { version_id: "102", version_name: "Second edition", size_bytes: 10, installed: false },
      { version_id: "101", version_name: "First edition", size_bytes: 10, installed: null },
    ]);
    await screen.findByText("Second edition");
    const rows = screen.getAllByRole("listitem");
    expect(within(rows[0]).getByRole("button", { name: "Installed" })).toBeDisabled();
    expect(within(rows[1]).getByText("Not installed")).toBeInTheDocument();
    expect(within(rows[2]).queryByText("Not installed")).not.toBeInTheDocument();
    expect(within(rows[2]).getByRole("button", { name: "Install this version" })).toBeEnabled();
    fireEvent.click(within(rows[1]).getByRole("button", { name: "Install this version" }));
    expect(choose).toHaveBeenCalledExactlyOnceWith("102");
  });
});
