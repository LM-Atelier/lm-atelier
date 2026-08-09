import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ReferenceDetail } from "./ReferenceDetail";
import { api } from "./api";
import type { ReferenceAsset, ReferenceSubject } from "./types";

vi.mock("./api", () => ({
  api: {
    referenceAssets: vi.fn(),
    attachReferenceAsset: vi.fn(),
    detachReferenceAsset: vi.fn(),
  },
}));

const mocked = vi.mocked(api);

const SUBJECT: ReferenceSubject = {
  id: "ref-1",
  name: "Ada Lovelace",
  mention_slug: "ada-lovelace",
  kind: "person",
  description: null,
  aliases_json: [],
  tags_json: [],
  cover_artifact_id: null,
  favorite: false,
  archived: false,
};

function asset(overrides: Partial<ReferenceAsset> = {}): ReferenceAsset {
  return {
    id: "asset-1",
    reference_subject_id: "ref-1",
    artifact_id: "art-1",
    caption: null,
    purpose: "identity",
    view_label: null,
    sort_order: 0,
    validation_state: "unchecked",
    ...overrides,
  };
}

function show(assets: ReferenceAsset[] = []) {
  mocked.referenceAssets.mockResolvedValue(assets);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ReferenceDetail subject={SUBJECT} onBack={() => {}} />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("reference detail", () => {
  it("shows what to type in a chat, not just the name", async () => {
    show();
    expect(await screen.findByText("@ada-lovelace")).toBeTruthy();
  });

  it("says an image is unchecked rather than implying it passed", async () => {
    show([asset()]);
    // Unchecked is not a synonym for usable. An image nobody has looked at must
    // not let the set claim a reviewed set's fidelity.
    expect(await screen.findByText("unchecked")).toBeTruthy();
  });

  it("keeps a near-duplicate image and says so, rather than refusing it", async () => {
    mocked.attachReferenceAsset.mockResolvedValue({
      asset: asset({ id: "asset-2", artifact_id: "art-2" }),
      similar: [
        { reference_asset_id: "asset-1", artifact_id: "art-1", mean_absolute_difference: 0.4 },
      ],
    });
    show([asset()]);

    fireEvent.change(await screen.findByLabelText("Artifact id to attach"), {
      target: { value: "art-2" },
    });
    fireEvent.click(screen.getByText("Add image"));

    // The wording has to make clear the image is in, not rejected: two close
    // shots are often deliberate and only the person adding them can judge.
    expect(await screen.findByText(/It was added anyway/)).toBeTruthy();
    await waitFor(() =>
      expect(mocked.attachReferenceAsset).toHaveBeenCalledWith("ref-1", {
        artifact_id: "art-2",
        purpose: "identity",
      }),
    );
  });

  it("says nothing when an image resembles nothing already held", async () => {
    mocked.attachReferenceAsset.mockResolvedValue({
      asset: asset({ id: "asset-2", artifact_id: "art-2" }),
      similar: [],
    });
    show([asset()]);

    fireEvent.change(await screen.findByLabelText("Artifact id to attach"), {
      target: { value: "art-2" },
    });
    fireEvent.click(screen.getByText("Add image"));

    await waitFor(() => expect(mocked.attachReferenceAsset).toHaveBeenCalled());
    expect(screen.queryByText(/It was added anyway/)).toBeNull();
  });

  it("surfaces a refusal instead of failing quietly", async () => {
    mocked.attachReferenceAsset.mockRejectedValue(
      new Error("Ada Lovelace already holds that exact image"),
    );
    show([asset()]);

    fireEvent.change(await screen.findByLabelText("Artifact id to attach"), {
      target: { value: "art-1" },
    });
    fireEvent.click(screen.getByText("Add image"));

    expect(await screen.findByText(/already holds that exact image/)).toBeTruthy();
  });

  it("removes only the membership when an image is detached", async () => {
    mocked.detachReferenceAsset.mockResolvedValue(undefined);
    show([asset()]);

    fireEvent.click(await screen.findByLabelText("Remove image 1"));
    await waitFor(() =>
      expect(mocked.detachReferenceAsset).toHaveBeenCalledWith("ref-1", "asset-1"),
    );
  });
});
