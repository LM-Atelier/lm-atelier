import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ReferencesLibrary } from "./ReferencesLibrary";
import { api } from "./api";
import type { ReferenceSubject } from "./types";

vi.mock("./api", () => ({
  api: {
    references: vi.fn(),
    createReference: vi.fn(),
    updateReference: vi.fn(),
    referenceDeletionImpact: vi.fn(),
    deleteReference: vi.fn(),
  },
}));

const mocked = vi.mocked(api);

function subject(overrides: Partial<ReferenceSubject> = {}): ReferenceSubject {
  return {
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
    ...overrides,
  };
}

function show(items: ReferenceSubject[] = [subject()]) {
  mocked.references.mockResolvedValue({ items, total: items.length, limit: 50, offset: 0 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ReferencesLibrary />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("references library", () => {
  it("shows the mention beside the name rather than implying it", async () => {
    // A rename does not move the mention by default, so the two can legitimately
    // disagree. Showing only the name would leave no way to know what to type.
    show([subject({ name: "Grace Hopper", mention_slug: "ada-lovelace" })]);

    expect(await screen.findByText("Grace Hopper")).toBeTruthy();
    expect(screen.getByText("@ada-lovelace")).toBeTruthy();
  });

  it("keeps an archived reference visible but receded", async () => {
    show([subject({ archived: true })]);

    const row = (await screen.findByText("Ada Lovelace")).closest("li");
    // Archiving is a removal from view, not from the record - so it is still
    // listed when asked for, and still readable when it is.
    expect(row?.className).toContain("archived");
  });

  it("asks what deletion would destroy before offering to do it", async () => {
    mocked.referenceDeletionImpact.mockResolvedValue({
      reference_subject_id: "ref-1",
      name: "Ada Lovelace",
      asset_count: 3,
      exclusive_artifact_ids: ["art-1"],
    });
    show();

    fireEvent.click(await screen.findByLabelText("Delete permanently"));

    await waitFor(() => expect(mocked.referenceDeletionImpact).toHaveBeenCalledWith("ref-1"));
    expect(await screen.findByText(/holds 3 image/)).toBeTruthy();
    expect(screen.getByText(/1 image\(s\) are used only here/)).toBeTruthy();
    // Nothing is deleted by asking.
    expect(mocked.deleteReference).not.toHaveBeenCalled();
  });

  it("deletes only against the impact it showed", async () => {
    mocked.referenceDeletionImpact.mockResolvedValue({
      reference_subject_id: "ref-1",
      name: "Ada Lovelace",
      asset_count: 2,
      exclusive_artifact_ids: [],
    });
    mocked.deleteReference.mockResolvedValue(undefined);
    show();

    fireEvent.click(await screen.findByLabelText("Delete permanently"));
    fireEvent.click(await screen.findByText("Delete permanently", { selector: "button.danger" }));

    // The count travels with the request: the server refuses to destroy
    // something other than what the user was looking at.
    await waitFor(() => expect(mocked.deleteReference).toHaveBeenCalledWith("ref-1", 2));
  });

  it("says an image shared with another reference would not be lost", async () => {
    mocked.referenceDeletionImpact.mockResolvedValue({
      reference_subject_id: "ref-1",
      name: "Ada Lovelace",
      asset_count: 4,
      exclusive_artifact_ids: [],
    });
    show();

    fireEvent.click(await screen.findByLabelText("Delete permanently"));
    expect(await screen.findByText(/No images would be lost/)).toBeTruthy();
  });

  it("offers only kinds the server accepts", async () => {
    show([]);

    fireEvent.click(await screen.findByText("New reference"));
    const options = [...screen.getByRole("combobox").querySelectorAll("option")].map(
      (option) => option.textContent,
    );
    // A workflow declares which kinds it can condition on, so an invented one
    // could never be matched against anything.
    expect(options).toContain("person");
    expect(options).not.toContain("spaceship");
  });
});
