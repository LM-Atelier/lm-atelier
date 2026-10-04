/** Checking a record file against this installation, from the model library. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { GenerationRecordCheck } from "./GenerationRecordCheck";
import { readGenerationRecordCheck } from "./generationRecord";

vi.mock("./api", () => ({
  api: {
    checkGenerationRecord: vi.fn(),
    planGenerationReplay: vi.fn(),
    createChat: vi.fn(),
    deleteChat: vi.fn(),
    replayGenerationRecord: vi.fn(),
    openEncryptedGenerationRecord: vi.fn(),
  },
}));

const checkRecord = vi.mocked(api.checkGenerationRecord);
const planReplay = vi.mocked(api.planGenerationReplay);
const replay = vi.mocked(api.replayGenerationRecord);
const openEncrypted = vi.mocked(api.openEncryptedGenerationRecord);
const ENCRYPTED_HEAD = [0x4c, 0x4d, 0x41, 0x41, 0x52, 0x43, 0x48, 0x00];

function answer(overrides: Record<string, unknown> = {}) {
  return {
    digest: `sha256:${"d".repeat(64)}`,
    operation: "text_to_image",
    requirements: [
      { kind: "workflow", sha256: "1".repeat(64), role: null, state: "present" },
      { kind: "model_file", sha256: "2".repeat(64), role: null, state: "inactive" },
      { kind: "lora", sha256: "3".repeat(64), role: null, state: "missing" },
    ],
    all_present: false,
    reproducibility: { missing: ["frozen_snapshot_absent"], status: "incomplete" },
    not_recorded: ["executed_graph_sha256", "runtime_version"],
    ...overrides,
  };
}

function renderCheck() {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><GenerationRecordCheck /></QueryClientProvider>);
}

function choose(content: string) {
  const file = new File([content], "generation-record.json", { type: "application/json" });
  fireEvent.change(screen.getByLabelText("Generation record file"), { target: { files: [file] } });
}

beforeEach(() => {
  checkRecord.mockReset();
  planReplay.mockReset();
  replay.mockReset();
  openEncrypted.mockReset();
  planReplay.mockResolvedValue({
    digest: `sha256:${"d".repeat(64)}`,
    operation: "text_to_image",
    ready: true,
    refusals: [],
  });
});
afterEach(cleanup);

describe("checking a record", () => {
  it("sends the file's own bytes and shows each requirement's state", async () => {
    checkRecord.mockResolvedValue(answer());
    renderCheck();

    choose('{"schema":"lm-atelier-output-recipe-v1"}');

    expect(await screen.findByText("Here and ready")).toBeInTheDocument();
    expect(screen.getByText("Here, not ready")).toBeInTheDocument();
    expect(screen.getByText("Not here")).toBeInTheDocument();
    expect(screen.getByText(/Nothing was installed or changed/)).toBeInTheDocument();
    expect(screen.getByText("This generation did not keep a frozen copy of its inputs.")).toBeInTheDocument();
    const sent = checkRecord.mock.calls[0][0];
    expect(new TextDecoder().decode(sent)).toBe('{"schema":"lm-atelier-output-recipe-v1"}');
  });

  it("says plainly when everything is here", async () => {
    checkRecord.mockResolvedValue(answer({
      requirements: [{ kind: "workflow", sha256: "1".repeat(64), role: null, state: "present" }],
      all_present: true,
      reproducibility: { missing: [], status: "recorded" },
    }));
    renderCheck();

    choose("{}");

    expect(await screen.findByText(/Everything this record names is here and ready/)).toBeInTheDocument();
  });

  it("shows a fixed sentence for a file that is not a record, never the server's words", async () => {
    checkRecord.mockRejectedValue(Object.assign(new Error("server words"), { code: "output-recipe-unreadable" }));
    renderCheck();

    choose("not a record");

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This file is not a generation record this version can read.",
    );
    expect(screen.queryByText("server words")).toBeNull();
  });

  it("refuses an answer it does not recognise rather than showing part of it", async () => {
    checkRecord.mockResolvedValue(answer({
      requirements: [{ kind: "workflow", sha256: "not-a-hash", role: null, state: "present" }],
    }));
    renderCheck();

    choose("{}");

    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(screen.queryByText("Here and ready")).toBeNull();
  });

  it("takes a bundle as well, and says it holds a copy of the picture", async () => {
    checkRecord.mockResolvedValue(answer({
      picture: { sha256: "4".repeat(64), copy_of: "5".repeat(64), width: 1024, height: 768 },
    }));
    renderCheck();

    expect(screen.getByLabelText("Generation record file")).toHaveAttribute(
      "accept", "application/json,.json,application/zip,.zip,.encrypted,application/octet-stream",
    );
    choose("PK");

    expect(await screen.findByText(/a copy of the picture, 1024 × 768 pixels/)).toBeInTheDocument();
    expect(screen.getByText("Here and ready")).toBeInTheDocument();
  });

  it("names a bundle it cannot read as a bundle", async () => {
    checkRecord.mockRejectedValue(Object.assign(new Error("server words"), { code: "output-recipe-bundle-unreadable" }));
    renderCheck();

    choose("PK");

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This file is not a generation record bundle this version can read.",
    );
  });

  it("starts a ready record in a new chat and opens that chat", async () => {
    checkRecord.mockResolvedValue(answer());
    vi.mocked(api.createChat).mockResolvedValue({ id: "chat_again" } as Awaited<ReturnType<typeof api.createChat>>);
    replay.mockResolvedValue({});
    const onOpenChat = vi.fn();
    const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    render(<QueryClientProvider client={client}><GenerationRecordCheck onOpenChat={onOpenChat} /></QueryClientProvider>);

    choose("{}");
    fireEvent.click(await screen.findByRole("button", { name: "Generate again" }));

    await waitFor(() => expect(onOpenChat).toHaveBeenCalledWith("chat_again"));
    expect(planReplay.mock.calls[0][0]).toBe(checkRecord.mock.calls[0][0]);
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("offers no replay where nothing could open the chat", async () => {
    checkRecord.mockResolvedValue(answer());
    renderCheck();

    choose("{}");

    await screen.findByText("Here and ready");
    expect(screen.queryByRole("button", { name: "Generate again" })).toBeNull();
  });

  it("closes and is ready for another file", async () => {
    checkRecord.mockResolvedValue(answer());
    renderCheck();
    choose("{}");
    await screen.findByText("Here and ready");

    fireEvent.click(screen.getByRole("button", { name: "Close the record check" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    choose("{}");
    expect(await screen.findByText("Here and ready")).toBeInTheDocument();
    expect(checkRecord).toHaveBeenCalledTimes(2);
  });
});

describe("reading the answer", () => {
  it("does not call a check all present unless every requirement is", () => {
    const read = readGenerationRecordCheck(answer({ all_present: true }));

    expect(read.allPresent).toBe(false);
    expect(read.requirements.map((item) => item.state)).toEqual(["present", "inactive", "missing"]);
  });

  it("reads a bare record's answer as holding no picture", () => {
    expect(readGenerationRecordCheck(answer()).picture).toBeNull();
    expect(readGenerationRecordCheck(answer({ picture: null })).picture).toBeNull();
  });

  it("refuses a picture it cannot vouch for", () => {
    for (const picture of [
      { sha256: "4".repeat(64), copy_of: "5".repeat(64), width: 0, height: 768 },
      { sha256: "4".repeat(64), copy_of: "5".repeat(64), width: 10.5, height: 768 },
      { sha256: "4".repeat(64), copy_of: "5".repeat(64), width: 10, height: 0 },
      { sha256: "not-a-hash", copy_of: "5".repeat(64), width: 10, height: 10 },
      { sha256: "4".repeat(64), copy_of: "not-a-hash", width: 10, height: 10 },
      "a picture",
    ]) {
      expect(() => readGenerationRecordCheck(answer({ picture }))).toThrow("malformed");
    }
  });
});

describe("the inputs a bundle carries", () => {
  const inputs = [
    { kind: "input", sha256: "6".repeat(64), role: "source", state: "missing" },
    { kind: "input", sha256: "7".repeat(64), role: "mask", state: "present" },
  ];

  it("reads which recorded inputs the checked file carries a copy of", () => {
    expect(readGenerationRecordCheck(answer({ requirements: inputs, bundled_inputs: [0, 1] })).bundledInputs)
      .toEqual([0, 1]);
    expect(readGenerationRecordCheck(answer({ requirements: inputs, bundled_inputs: [] })).bundledInputs).toEqual([]);
    expect(readGenerationRecordCheck(answer()).bundledInputs).toEqual([]);
  });

  it("refuses positions the record does not have, out of order, or not whole", () => {
    for (const bundled of [[2], [1], [0, 0], [0.5], ["0"], "0"]) {
      expect(() => readGenerationRecordCheck(answer({ requirements: inputs, bundled_inputs: bundled })))
        .toThrow("malformed");
    }
  });
});

describe("checking an encrypted record", () => {
  function chooseEncrypted() {
    const file = new File([new Uint8Array([...ENCRYPTED_HEAD, 1, 2, 3])], "record.json.encrypted");
    fireEvent.change(screen.getByLabelText("Generation record file"), { target: { files: [file] } });
    return file;
  }

  it("asks for its passphrase, opens it and checks the record it holds", async () => {
    const plain = new TextEncoder().encode("{}").buffer;
    openEncrypted.mockResolvedValue(plain);
    checkRecord.mockResolvedValue(answer());
    renderCheck();
    chooseEncrypted();

    expect(await screen.findByText("record.json.encrypted is encrypted. Enter its passphrase to check it.")).toBeVisible();
    const field = screen.getByLabelText("Record passphrase");
    await waitFor(() => expect(field).toHaveFocus());
    expect(checkRecord).not.toHaveBeenCalled();
    fireEvent.change(field, { target: { value: "correct horse" } });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() => expect(checkRecord).toHaveBeenCalledWith(plain));
    const [sent, passphrase] = openEncrypted.mock.calls[0];
    expect(new Uint8Array(sent).slice(0, 8)).toEqual(new Uint8Array(ENCRYPTED_HEAD));
    expect(passphrase).toBe("correct horse");
    expect(await screen.findByText("Here and ready")).toBeVisible();
    expect(screen.queryByLabelText("Record passphrase")).toBeNull();
  });

  it("keeps asking after a passphrase that does not open it, and checks nothing", async () => {
    openEncrypted.mockRejectedValue(new Error("The passphrase is wrong, or the archive is damaged."));
    renderCheck();
    chooseEncrypted();
    fireEvent.change(await screen.findByLabelText("Record passphrase"), { target: { value: "wrong" } });

    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("The passphrase is wrong, or the archive is damaged.");
    expect(screen.getByLabelText("Record passphrase")).toBeVisible();
    expect(checkRecord).not.toHaveBeenCalled();
  });

  it("never applies an open the person closed and replaced with another file", async () => {
    let finish: (value: ArrayBuffer) => void = () => undefined;
    openEncrypted.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    checkRecord.mockResolvedValue(answer());
    renderCheck();
    chooseEncrypted();
    fireEvent.change(await screen.findByLabelText("Record passphrase"), { target: { value: "correct horse" } });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    await waitFor(() => expect(openEncrypted).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: "Close the record check" }));
    choose('{"schema":"lm-atelier-output-recipe-v1"}');
    await waitFor(() => expect(checkRecord).toHaveBeenCalledTimes(1));
    await act(async () => finish(new TextEncoder().encode("{\"earlier\":true}").buffer));

    expect(checkRecord).toHaveBeenCalledTimes(1);
    expect(new TextDecoder().decode(checkRecord.mock.calls[0][0])).toBe('{"schema":"lm-atelier-output-recipe-v1"}');
  });

  it("keeps focus inside the dialog once the opened record is being checked", async () => {
    openEncrypted.mockResolvedValue(new TextEncoder().encode("{}").buffer);
    checkRecord.mockResolvedValue(answer());
    renderCheck();
    chooseEncrypted();
    const field = await screen.findByLabelText("Record passphrase");
    fireEvent.change(field, { target: { value: "correct horse" } });

    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() => expect(screen.queryByLabelText("Record passphrase")).toBeNull());
    expect(screen.getByRole("dialog").contains(document.activeElement)).toBe(true);
  });
});
