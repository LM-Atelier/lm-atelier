import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ReferenceDetail } from "./ReferenceDetail";
import { api } from "./api";
import type { ArtifactLibraryItem, ReferenceAsset, ReferenceAssetReviewed, ReferenceSubject } from "./types";

vi.mock("./api", () => ({
  api: {
    referenceAssets: vi.fn(),
    attachReferenceAsset: vi.fn(),
    detachReferenceAsset: vi.fn(),
    artifacts: vi.fn(),
    setReferenceCover: vi.fn(),
    clearReferenceCover: vi.fn(),
    updateReference: vi.fn(),
    reviewReferenceAsset: vi.fn(),
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

function libraryItem(id: string): ArtifactLibraryItem {
  return {
    id,
    sha256: `sha-${id}`,
    kind: "image",
    media_type: "image/png",
    size_bytes: 1024,
    original_name: `${id}.png`,
    metadata_json: {},
    created_at: "2026-01-01T00:00:00Z",
    reference_count: 0,
    chat_ids: [],
    project_ids: [],
  };
}

function show(assets: ReferenceAsset[] = [], library = [libraryItem("art-2"), libraryItem("art-3")]) {
  mocked.referenceAssets.mockResolvedValue(assets);
  mocked.artifacts.mockResolvedValue(library);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ReferenceDetail subject={SUBJECT} onBack={() => {}} />
    </QueryClientProvider>,
  );
}

/** Open the picker, choose images by their library name, and confirm. */
async function pick(names: string[]) {
  fireEvent.click(await screen.findByText("Add images"));
  for (const name of names) {
    fireEvent.click(await screen.findByLabelText(name));
  }
  fireEvent.click(screen.getByRole("button", { name: `Add ${names.length}` }));
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("reference image review", () => {
  it("records a deliberate usable decision and shows the settled response", async () => {
    mocked.reviewReferenceAsset.mockResolvedValue({
      asset: asset({ validation_state: "usable" }), width: 512, height: 512, review_version: 2,
    });
    show([asset()]);
    fireEvent.click(await screen.findByRole("button", { name: "Review image 1" }));
    const save = screen.getByRole("button", { name: "Save review" });
    expect(save.hasAttribute("disabled")).toBe(true);
    expect(mocked.reviewReferenceAsset).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Review outcome"), { target: { value: "usable" } });
    fireEvent.click(save);
    await waitFor(() => expect(mocked.reviewReferenceAsset).toHaveBeenCalledWith(
      "ref-1", "asset-1", { outcome: "usable", reasons: [] },
    ));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByText("usable")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Review image 1" })).toBeNull();
  });

  it.each(["weak", "rejected"] as const)("requires a reason for a %s decision", async (outcome) => {
    mocked.reviewReferenceAsset.mockResolvedValue({
      asset: asset({ validation_state: outcome }), width: 512, height: 512, review_version: 2,
    });
    show([asset()]);
    fireEvent.click(await screen.findByRole("button", { name: "Review image 1" }));
    fireEvent.change(screen.getByLabelText("Review outcome"), { target: { value: outcome } });
    const save = screen.getByRole("button", { name: "Save review" });
    expect(save.hasAttribute("disabled")).toBe(true);
    fireEvent.change(screen.getByLabelText("Review reason"), { target: { value: "   " } });
    expect(save.hasAttribute("disabled")).toBe(true);
    fireEvent.change(screen.getByLabelText("Review reason"), { target: { value: "  Image is blurred  " } });
    fireEvent.click(save);
    await waitFor(() => expect(mocked.reviewReferenceAsset).toHaveBeenCalledWith(
      "ref-1", "asset-1", { outcome, reasons: ["Image is blurred"] },
    ));
  });

  it("keeps a refused review open for correction and retry", async () => {
    mocked.reviewReferenceAsset.mockRejectedValueOnce(new Error("The image could not be verified."));
    mocked.reviewReferenceAsset.mockResolvedValueOnce({
      asset: asset({ validation_state: "weak" }), width: 512, height: 512, review_version: 2,
    });
    show([asset()]);
    fireEvent.click(await screen.findByRole("button", { name: "Review image 1" }));
    fireEvent.change(screen.getByLabelText("Review outcome"), { target: { value: "weak" } });
    fireEvent.change(screen.getByLabelText("Review reason"), { target: { value: "Image is blurred" } });
    fireEvent.click(screen.getByRole("button", { name: "Save review" }));
    expect(await screen.findByText("The image could not be verified.")).toBeTruthy();
    expect(screen.getByText("unchecked")).toBeTruthy();
    expect((screen.getByLabelText("Review reason") as HTMLTextAreaElement).value).toBe("Image is blurred");
    fireEvent.click(screen.getByRole("button", { name: "Save review" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(mocked.reviewReferenceAsset).toHaveBeenCalledTimes(2);
    expect(screen.getByText("weak")).toBeTruthy();
  });

  it("does not offer another decision for settled images", async () => {
    show([asset({ validation_state: "usable" }), asset({ id: "second", sort_order: 1, validation_state: "rejected" })]);
    expect(await screen.findByText("usable")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Review image/ })).toBeNull();
  });

  it("cancels a review without sending it", async () => {
    show([asset()]);
    fireEvent.click(await screen.findByRole("button", { name: "Review image 1" }));
    fireEvent.change(screen.getByLabelText("Review outcome"), { target: { value: "usable" } });
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(mocked.reviewReferenceAsset).not.toHaveBeenCalled();
  });

  it("keeps a pending decision bound to the selected image and submits it once", async () => {
    let finish!: (result: ReferenceAssetReviewed) => void;
    mocked.reviewReferenceAsset.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    const second = asset({ id: "second", artifact_id: "second-image", sort_order: 1 });
    show([asset(), second]);
    fireEvent.click(await screen.findByRole("button", { name: "Review image 2" }));
    fireEvent.change(screen.getByLabelText("Review outcome"), { target: { value: "usable" } });
    fireEvent.click(screen.getByRole("button", { name: "Save review" }));
    const pending = await screen.findByRole("button", { name: "Saving review…" });
    expect(pending.hasAttribute("disabled")).toBe(true);
    expect(screen.getByLabelText("Review outcome").hasAttribute("disabled")).toBe(true);
    expect(screen.getByRole("button", { name: "Cancel" }).hasAttribute("disabled")).toBe(true);
    fireEvent.click(pending);
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(screen.getByRole("dialog")).toBeTruthy();
    expect(mocked.reviewReferenceAsset).toHaveBeenCalledTimes(1);
    expect(mocked.reviewReferenceAsset).toHaveBeenCalledWith("ref-1", "second", { outcome: "usable", reasons: [] });
    finish({ asset: { ...second, validation_state: "usable" }, width: 512, height: 512, review_version: 2 });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByRole("button", { name: "Review image 1" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Review image 2" })).toBeNull();
  });
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

  it("attaches what was picked from the library, under the chosen purpose", async () => {
    mocked.attachReferenceAsset.mockResolvedValue({
      asset: asset({ id: "asset-2", artifact_id: "art-2" }),
      similar: [],
    });
    show([asset()]);

    fireEvent.click(await screen.findByText("Add images"));
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "pose" } });
    fireEvent.click(await screen.findByLabelText("art-2.png"));
    fireEvent.click(screen.getByRole("button", { name: "Add 1" }));

    // The id never has to be typed, and the purpose chosen in the picker is the
    // one the image arrives with.
    await waitFor(() =>
      expect(mocked.attachReferenceAsset).toHaveBeenCalledWith("ref-1", {
        artifact_id: "art-2",
        purpose: "pose",
      }),
    );
  });

  it("keeps a near-duplicate image and says so, rather than refusing it", async () => {
    mocked.attachReferenceAsset.mockResolvedValue({
      asset: asset({ id: "asset-2", artifact_id: "art-2" }),
      similar: [
        { reference_asset_id: "asset-1", artifact_id: "art-1", mean_absolute_difference: 0.4 },
      ],
    });
    show([asset()]);

    await pick(["art-2.png"]);

    // The wording has to make clear the image is in, not rejected: two close
    // shots are often deliberate and only the person adding them can judge.
    expect(await screen.findByText(/It was added anyway/)).toBeTruthy();
  });

  it("says nothing when an image resembles nothing already held", async () => {
    mocked.attachReferenceAsset.mockResolvedValue({
      asset: asset({ id: "asset-2", artifact_id: "art-2" }),
      similar: [],
    });
    show([asset()]);

    await pick(["art-2.png"]);

    await waitFor(() => expect(mocked.attachReferenceAsset).toHaveBeenCalled());
    expect(screen.queryByText(/It was added anyway/)).toBeNull();
  });

  it("adds the images it can when one of them is refused", async () => {
    mocked.attachReferenceAsset.mockImplementation(async (_id, body) => {
      if (body.artifact_id === "art-2") {
        throw new Error("Ada Lovelace already holds that exact image");
      }
      return { asset: asset({ id: "asset-9", artifact_id: body.artifact_id }), similar: [] };
    });
    show([asset()]);

    await pick(["art-2.png", "art-3.png"]);

    // A refusal on one image is not a reason to drop the others, and the report
    // has to name what happened rather than fail the whole batch silently.
    expect(await screen.findByText(/already holds that exact image/)).toBeTruthy();
    await waitFor(() =>
      expect(mocked.attachReferenceAsset).toHaveBeenCalledWith("ref-1", {
        artifact_id: "art-3",
        purpose: "identity",
      }),
    );
  });

  it("removes only the membership when an image is detached", async () => {
    mocked.detachReferenceAsset.mockResolvedValue(undefined);
    show([asset()]);

    fireEvent.click(await screen.findByLabelText("Remove image 1"));
    await waitFor(() =>
      expect(mocked.detachReferenceAsset).toHaveBeenCalledWith("ref-1", "asset-1"),
    );
  });

  it("lets an image stand for the reference", async () => {
    mocked.setReferenceCover.mockResolvedValue({ ...SUBJECT, cover_artifact_id: "art-1" });
    show([asset()]);

    fireEvent.click(await screen.findByLabelText("Use image 1 for Ada Lovelace"));

    await waitFor(() =>
      expect(mocked.setReferenceCover).toHaveBeenCalledWith("ref-1", "art-1"),
    );
  });

  it("pressing the current cover clears it rather than removing the image", async () => {
    // There has to be a way back to no cover that is not "delete the picture".
    mocked.clearReferenceCover.mockResolvedValue({ ...SUBJECT, cover_artifact_id: null });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    mocked.referenceAssets.mockResolvedValue([asset()]);
    mocked.artifacts.mockResolvedValue([]);
    render(
      <QueryClientProvider client={client}>
        <ReferenceDetail
          subject={{ ...SUBJECT, cover_artifact_id: "art-1" }}
          onBack={() => {}}
        />
      </QueryClientProvider>,
    );

    fireEvent.click(
      await screen.findByLabelText("Stop image 1 standing for Ada Lovelace"),
    );

    await waitFor(() => expect(mocked.clearReferenceCover).toHaveBeenCalledWith("ref-1"));
    expect(mocked.detachReferenceAsset).not.toHaveBeenCalled();
  });

  it("splits the other names on save rather than while they are typed", async () => {
    // A list that rewrote itself mid-word would fight whoever is typing it, so
    // a half-finished "Countess Lovelace" is never briefly two names.
    mocked.updateReference.mockResolvedValue(SUBJECT);
    show();

    fireEvent.change(await screen.findByLabelText("Other names, separated by commas"), {
      target: { value: "Countess Lovelace, AAL" },
    });
    fireEvent.click(screen.getByText("Save details"));

    await waitFor(() =>
      expect(mocked.updateReference).toHaveBeenCalledWith("ref-1", {
        aliases: ["Countess Lovelace", "AAL"],
      }),
    );
  });

  it("adopts the canonical saved details and does not stay dirty", async () => {
    mocked.updateReference.mockResolvedValue({
      ...SUBJECT,
      aliases_json: ["Countess Lovelace"],
    });
    show();

    fireEvent.change(await screen.findByLabelText("Other names, separated by commas"), {
      target: { value: " Countess Lovelace, countess lovelace " },
    });
    fireEvent.click(screen.getByText("Save details"));

    await waitFor(() =>
      expect(screen.getByText("Save details").hasAttribute("disabled")).toBe(true),
    );
    expect(screen.getByLabelText("Other names, separated by commas")).toHaveProperty(
      "value",
      "Countess Lovelace",
    );
  });

  it("does not overwrite details that were not edited", async () => {
    mocked.updateReference.mockResolvedValue({ ...SUBJECT, description: "Mathematician" });
    show();

    fireEvent.change(await screen.findByLabelText("Description"), {
      target: { value: "Mathematician" },
    });
    fireEvent.click(screen.getByText("Save details"));

    await waitFor(() =>
      expect(mocked.updateReference).toHaveBeenCalledWith("ref-1", {
        description: "Mathematician",
      }),
    );
  });

  it("offers nothing to save until something is edited", async () => {
    show();

    const save = await screen.findByText("Save details");
    expect(save.hasAttribute("disabled")).toBe(true);

    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "Mathematician" },
    });
    expect(screen.getByText("Save details").hasAttribute("disabled")).toBe(false);
  });
});
