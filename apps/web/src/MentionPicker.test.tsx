import { useRef, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MessageField } from "./MessageField";
import { api } from "./api";
import type { ReferenceSubject } from "./types";
import type { TrackedMention } from "./mentionDraft";

vi.mock("./api", () => ({ api: { references: vi.fn() } }));
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

function Harness({ onMention }: { onMention?: (mention: TrackedMention) => void }) {
  const field = useRef<HTMLTextAreaElement | null>(null);
  const [text, setText] = useState("");
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={client}>
      <MessageField
        field={field}
        value={text}
        onChange={setText}
        onSubmit={() => {}}
        onMention={onMention}
      />
    </QueryClientProvider>
  );
}

function type(value: string) {
  const field = screen.getByLabelText("Message") as HTMLTextAreaElement;
  fireEvent.change(field, { target: { value } });
  // jsdom does not move a caret on its own, so say where it ended up.
  field.selectionStart = value.length;
  fireEvent.select(field);
  return field;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("mentions in the composer", () => {
  it("offers subjects once an @ is typed", async () => {
    mocked.references.mockResolvedValue({ items: [subject()], total: 1, limit: 50, offset: 0 });
    render(<Harness onMention={vi.fn()} />);

    type("draw @");

    expect(await screen.findByText("Ada Lovelace")).toBeTruthy();
  });

  it("attaches the chosen subject's id, not the typed text", async () => {
    // The whole point: the id comes from the choice. Matching prose against
    // names would bind whoever the text most resembles.
    mocked.references.mockResolvedValue({ items: [subject()], total: 1, limit: 50, offset: 0 });
    const onMention = vi.fn();
    render(<Harness onMention={onMention} />);

    type("draw @ad");
    fireEvent.mouseDown(await screen.findByText("Ada Lovelace"));

    expect(onMention).toHaveBeenCalledWith({
      referenceSubjectId: "ref-1",
      mentionSlug: "ada-lovelace",
    });
    expect((screen.getByLabelText("Message") as HTMLTextAreaElement).value).toBe(
      "draw @ada-lovelace ",
    );
  });

  it("attaches nothing when the characters are only typed", async () => {
    mocked.references.mockResolvedValue({ items: [subject()], total: 1, limit: 50, offset: 0 });
    const onMention = vi.fn();
    render(<Harness onMention={onMention} />);

    type("draw @ada-lovelace");

    await waitFor(() => expect(mocked.references).toHaveBeenCalled());
    expect(onMention).not.toHaveBeenCalled();
  });

  it("says which kind of empty it is", async () => {
    // "nothing matches what you typed" and "you have no references" need
    // different next actions from the reader.
    mocked.references.mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 });
    render(<Harness onMention={vi.fn()} />);

    type("draw @nobody");

    expect(await screen.findByText(/answers to @nobody/)).toBeTruthy();
  });

  it("closes on Escape without sending", async () => {
    mocked.references.mockResolvedValue({ items: [subject()], total: 1, limit: 50, offset: 0 });
    render(<Harness onMention={vi.fn()} />);

    const field = type("draw @");
    expect(await screen.findByText("Ada Lovelace")).toBeTruthy();
    fireEvent.keyDown(field, { key: "Escape" });

    await waitFor(() => expect(screen.queryByText("Ada Lovelace")).toBeNull());
  });

  it("stays out of the way when the caller wants no mentions", async () => {
    render(<Harness />);

    type("draw @");

    // No picker, and crucially no reference lookup at all.
    await waitFor(() => expect(mocked.references).not.toHaveBeenCalled());
  });
});
