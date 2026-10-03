import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { PictureRemixDialog } from "./PictureRemixDialog";
import { PictureFileSettings } from "./PictureFileSettings";
import { api } from "./api";
import { discardBlankChat } from "./discardBlankChat";
import { applicableClaims, readRemixPreview } from "./pictureRemix";

vi.mock("./discardBlankChat", () => ({ discardBlankChat: vi.fn() }));
vi.mock("./api", () => ({
  api: {
    workflowReadyRevisions: vi.fn(),
    profiles: vi.fn(),
    remixPreview: vi.fn(),
    pictureSettings: vi.fn(),
    createChat: vi.fn(),
    remixPicture: vi.fn(),
  },
}));
afterEach(() => { cleanup(); vi.resetAllMocks(); });

const artifactId = `sha256:${"e".repeat(64)}`;
const PROMPT = "a ceramic cup on a wooden table";

function claim(key: string, value: string | number, state: string, reason: string | null = null) {
  return { key, setting: key === "guidance" ? "cfg" : key, value, source: key, state, reason, applied: key === "prompt" };
}

const preview = {
  artifact_id: artifactId,
  metadata: { dialect: "parameters", parser_version: 1, budget_version: 1, digest: "sha256:1" },
  role: "words",
  workflow_revision_id: "rev_cup",
  profile_id: "profile_cup",
  operation: "text_to_image",
  source: null,
  claims: [
    claim("prompt", PROMPT, "supported"),
    claim("steps", 20, "supported"),
    claim("sampler", "Euler a", "unresolved", "no_vocabulary"),
    claim("seed", "6342567893452345234", "incompatible", "value_refused"),
    claim("denoise", 0.5, "ignored", "edit_only"),
    claim("width", 512, "supported"),
    claim("height", 768, "supported"),
  ],
  shape: null,
  ignored: [{ name: "Model", reason: "names_a_file" }],
  resolved: {
    text: PROMPT,
    settings: { negative_prompt: "", steps: 9, width: 1024, height: 1024, batch_size: 1 },
    seed_drawn: true,
    trigger_words: [],
    engine_prompt: PROMPT,
    strength: null,
  },
  refusals: [],
  ready: true,
  review_digest: "sha256:2",
};

const WORKFLOW = "Cups - Cup workflow · v1";

function renderDialog(onOpenChat?: (chatId: string) => void, onClose: () => void = () => {}) {
  vi.mocked(api.workflowReadyRevisions).mockResolvedValue([
    { family_id: "f", family_name: "Cups", workflow_id: "w", workflow_name: "Cup workflow",
      revision_id: "rev_cup", revision_version: 1, operation: "text_to_image" },
  ]);
  vi.mocked(api.profiles).mockResolvedValue([
    { id: "profile_cup", model_install_id: null, name: "Cup model", use_case: "", role: "image",
      engine: "mock", load_settings_json: {}, request_settings_json: {}, is_default: false },
    { id: "profile_video", model_install_id: null, name: "Moving model", use_case: "", role: "video",
      engine: "mock", load_settings_json: {}, request_settings_json: {}, is_default: false },
  ]);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><PictureRemixDialog artifactId={artifactId} onClose={onClose} onOpenChat={onOpenChat} /></QueryClientProvider>);
}

async function choose() {
  // A choice can be made only once its options are there to choose from.
  await screen.findByRole("option", { name: WORKFLOW });
  await screen.findByRole("option", { name: "Cup model" });
  fireEvent.change(screen.getByLabelText("Workflow"), { target: { value: "rev_cup" } });
  fireEvent.change(screen.getByLabelText("Model"), { target: { value: "profile_cup" } });
}

describe("remix preview", () => {
  it("checks the picture's settings only once a workflow and a model are chosen", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(preview);
    renderDialog();
    await screen.findByRole("option", { name: WORKFLOW });
    expect(screen.queryByRole("option", { name: "Moving model" })).toBeNull();
    expect(api.remixPreview).not.toHaveBeenCalled();

    await choose();

    await screen.findByText("A remix would make one picture with");
    expect(api.remixPreview).toHaveBeenCalledWith(
      artifactId,
      { workflow_revision_id: "rev_cup", profile_id: "profile_cup", apply: [], role: "words" },
      expect.any(AbortSignal),
    );
    const items = screen.getAllByRole("listitem").map((item) => item.textContent);
    expect(items).toContain("Sampler: Euler a. It cannot be checked: this workflow does not list the names it knows, so this one cannot be checked.");
    expect(items).toContain("Seed: 6342567893452345234. It cannot be used: this workflow does not take this value.");
    expect(items).toContain("Denoise: 0.5. It is not used: only for changing a picture, not making one from words.");
    expect(screen.getByText("chosen when it is made")).toBeTruthy();
    expect(screen.getByText("none")).toBeTruthy();
  });

  it("offers only what can be used as it is, a size as one choice, and checks again with it", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(preview);
    renderDialog();
    await choose();
    await screen.findByText("Use from the file");

    const offered = screen.getAllByRole("checkbox").map((box) => box.closest("label")?.textContent);
    expect(offered).toEqual(["Steps", "Size"]);
    fireEvent.click(screen.getByRole("checkbox", { name: "Size" }));

    await waitFor(() => expect(api.remixPreview).toHaveBeenLastCalledWith(
      artifactId,
      { workflow_revision_id: "rev_cup", profile_id: "profile_cup", apply: ["width", "height"], role: "words" },
      expect.any(AbortSignal),
    ));
  });

  it("starts what is applied over when the workflow or model changes", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(preview);
    renderDialog();
    await choose();
    fireEvent.click(await screen.findByRole("checkbox", { name: "Steps" }));
    await waitFor(() => expect(api.remixPreview).toHaveBeenLastCalledWith(
      artifactId, expect.objectContaining({ apply: ["steps"] }), expect.any(AbortSignal),
    ));

    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "profile_cup" } });

    await waitFor(() => expect(api.remixPreview).toHaveBeenLastCalledWith(
      artifactId, expect.objectContaining({ apply: [] }), expect.any(AbortSignal),
    ));
  });

  it("says why a choice cannot be used", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue({
      ...preview, ready: false, claims: [], resolved: null, review_digest: null,
      refusals: [{ code: "remix-model-unusable", message: "This model cannot run with this workflow." }],
    });
    renderDialog();
    await choose();

    expect(await screen.findByRole("alert")).toHaveTextContent("This model cannot run with this workflow.");
    expect(screen.queryByText("A remix would make one picture with")).toBeNull();
  });

  it.each([
    { ...preview, ready: "yes" },
    { ...preview, claims: [{ ...preview.claims[1], value: 2 ** 60 }] },
    { ...preview, claims: [{ ...preview.claims[1], state: "maybe" }] },
    { ...preview, shape: { width: 0, height: 768 } },
    { ...preview, shape: undefined },
    { ...preview, role: "other" },
    { ...preview, resolved: { ...preview.resolved, strength: { parameter: "denoise", mode: "high", value: 1, from_file: false } } },
  ])("shows nothing from an answer of another shape", async (other) => {
    vi.mocked(api.remixPreview).mockResolvedValue(other);
    renderDialog();
    await choose();

    expect(await screen.findByRole("alert")).toHaveTextContent("These choices could not be checked.");
  });
});

describe("making a remix", () => {
  it("keeps a ticked box and its focus while the new answer is checked", async () => {
    let answer: (value: unknown) => void = () => {};
    vi.mocked(api.remixPreview)
      .mockResolvedValueOnce(preview)
      .mockImplementationOnce(() => new Promise((resolve) => { answer = resolve; }));
    renderDialog();
    await choose();

    const steps = await screen.findByRole("checkbox", { name: "Steps" });
    steps.focus();
    fireEvent.click(steps);

    await waitFor(() => expect(api.remixPreview).toHaveBeenCalledTimes(2));
    const busy = screen.getByRole("checkbox", { name: "Steps" });
    expect(busy).toBe(steps);
    expect(document.activeElement).toBe(steps);
    expect(busy).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("button", { name: "Make this picture" })).toHaveAttribute("aria-disabled", "true");
    answer({ ...preview, review_digest: "sha256:3" });
    await waitFor(() => expect(screen.getByRole("button", { name: "Make this picture" }))
      .toHaveAttribute("aria-disabled", "false"));
    expect(document.activeElement).toBe(steps);
  });

  it("makes the previewed picture in a new chat and opens it", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(preview);
    vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as never);
    vi.mocked(api.remixPicture).mockResolvedValue({});
    const opened = vi.fn();
    const closed = vi.fn();
    renderDialog(opened, closed);
    await choose();
    fireEvent.click(await screen.findByRole("checkbox", { name: "Size" }));
    await waitFor(() => expect(api.remixPreview).toHaveBeenLastCalledWith(
      artifactId, expect.objectContaining({ apply: ["width", "height"] }), expect.any(AbortSignal),
    ));
    const make = screen.getByRole("button", { name: "Make this picture" });
    await waitFor(() => expect(make).toHaveAttribute("aria-disabled", "false"));

    fireEvent.click(make);

    await waitFor(() => expect(opened).toHaveBeenCalledWith("chat_new"));
    expect(api.remixPicture).toHaveBeenCalledWith("chat_new", {
      artifact_id: artifactId,
      workflow_revision_id: "rev_cup",
      profile_id: "profile_cup",
      apply: ["width", "height"],
      role: "words",
      review_digest: "sha256:2",
    });
    expect(closed).toHaveBeenCalled();
    expect(discardBlankChat).not.toHaveBeenCalled();
  });

  it("removes the new chat and says why when the remix is refused", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(preview);
    vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as never);
    vi.mocked(discardBlankChat).mockResolvedValue(undefined);
    vi.mocked(api.remixPicture).mockRejectedValue(
      Object.assign(new Error("changed"), { code: "remix-review-changed" }),
    );
    const opened = vi.fn();
    renderDialog(opened);
    await choose();
    const make = await screen.findByRole("button", { name: "Make this picture" });
    await waitFor(() => expect(make).toHaveAttribute("aria-disabled", "false"));

    fireEvent.click(make);

    expect(await screen.findByText(
      "What this remix would run changed since it was shown. It has been checked again.",
    )).toBeTruthy();
    expect(discardBlankChat).toHaveBeenCalledWith("chat_new");
    expect(opened).not.toHaveBeenCalled();
  });

  it("cannot be made until a ready answer is shown", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue({
      ...preview, ready: false, claims: [], resolved: null, review_digest: null,
      refusals: [{ code: "remix-model-unusable", message: "This model cannot run with this workflow." }],
    });
    renderDialog();
    const make = await screen.findByRole("button", { name: "Make this picture" });
    expect(make).toHaveAttribute("aria-disabled", "true");
    await choose();
    await screen.findByRole("alert");

    fireEvent.click(make);

    expect(make).toHaveAttribute("aria-disabled", "true");
    expect(api.createChat).not.toHaveBeenCalled();
  });
});

describe("what a remix shows and keeps current", () => {
  it("says a seed is drawn only when one is", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue({
      ...preview, resolved: { ...preview.resolved, seed_drawn: false },
    });
    renderDialog();
    await choose();

    await screen.findByText("A remix would make one picture with");
    expect(screen.queryByText("chosen when it is made")).toBeNull();
    const seed = screen.getByText("Seed", { selector: "dt" });
    expect(seed.nextElementSibling?.textContent).toBe("the workflow's own");
  });

  it("names a chosen workflow that is no longer listed rather than showing none", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(preview);
    renderDialog();
    vi.mocked(api.workflowReadyRevisions).mockImplementation(async (options = {}) => (
      options.revisionIds ? [] : [
        { family_id: "f", family_name: "Cups", workflow_id: "w", workflow_name: "Cup workflow",
          revision_id: "rev_cup", revision_version: 1, operation: "text_to_image" },
      ]
    ));
    await choose();

    const workflow = screen.getByLabelText("Workflow") as HTMLSelectElement;
    await waitFor(() => expect(workflow.value).toBe("rev_cup"));
    expect(workflow.selectedOptions[0]?.textContent).not.toBe("Choose a workflow");
  });

  it("keeps the choices while a remix is being started", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(preview);
    vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as never);
    vi.mocked(api.remixPicture).mockImplementation(() => new Promise(() => {}));
    renderDialog(vi.fn());
    await choose();
    const make = await screen.findByRole("button", { name: "Make this picture" });
    await waitFor(() => expect(make).toHaveAttribute("aria-disabled", "false"));

    fireEvent.click(make);
    await screen.findByText("Starting it in a new chat…");
    const model = screen.getByLabelText("Model") as HTMLSelectElement;
    fireEvent.change(model, { target: { value: "" } });

    expect(model.value).toBe("profile_cup");
    expect(model).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Starting it in a new chat…")).toBeTruthy();
  });

  it("cannot make a picture from an answer whose check then failed", async () => {
    vi.mocked(api.remixPreview)
      .mockResolvedValueOnce(preview)
      .mockRejectedValue(Object.assign(new Error("gone"), { code: "artifact-not-found" }));
    vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as never);
    vi.mocked(discardBlankChat).mockResolvedValue(undefined);
    vi.mocked(api.remixPicture).mockRejectedValue(
      Object.assign(new Error("gone"), { code: "artifact-not-found" }),
    );
    renderDialog();
    await choose();
    const make = await screen.findByRole("button", { name: "Make this picture" });
    await waitFor(() => expect(make).toHaveAttribute("aria-disabled", "false"));

    fireEvent.click(make);

    await screen.findByText("These choices could not be checked.");
    expect(make).toHaveAttribute("aria-disabled", "true");
  });
});

describe("remix from the settings in a picture's file", () => {
  function renderSettings(claims: { key: string; value: string | number; source: string }[]) {
    vi.mocked(api.pictureSettings).mockResolvedValue({
      dialect: "parameters", parser_version: 1, budget_version: 1, digest: "sha256:1",
      claims, ignored: [], warnings: [],
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><PictureFileSettings artifactId={artifactId} /></QueryClientProvider>);
    const details = screen.getByText("Settings in the file", { selector: "summary" }).parentElement as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
  }

  it("is offered for a file with a prompt and opens over the page", async () => {
    vi.mocked(api.workflowReadyRevisions).mockResolvedValue([]);
    vi.mocked(api.profiles).mockResolvedValue([]);
    renderSettings([{ key: "prompt", value: PROMPT, source: "prompt" }]);

    fireEvent.click(await screen.findByRole("button", { name: "Remix these settings" }));

    const dialog = await screen.findByRole("dialog");
    expect(dialog.closest("details")).toBeNull();
  });

  it("is not offered for a file with no prompt", async () => {
    renderSettings([{ key: "steps", value: 20, source: "Steps" }]);

    await screen.findByText("Steps");
    expect(screen.queryByRole("button", { name: "Remix these settings" })).toBeNull();
  });
});

describe("remix preview answers", () => {
  it("keeps width and height together as one size", () => {
    expect(applicableClaims(readRemixPreview(preview).claims)).toEqual(["steps", "size"]);
    expect(applicableClaims(readRemixPreview({ ...preview, claims: [preview.claims[5]] }).claims)).toEqual([]);
  });
});

describe("a picture whose own size this workflow cannot make", () => {
  const shaped = {
    ...preview,
    claims: [
      claim("prompt", PROMPT, "supported"),
      claim("width", 1000, "incompatible", "value_refused"),
      claim("height", 1500, "incompatible", "value_refused"),
    ],
    shape: { width: 768, height: 1152 },
    resolved: { ...preview.resolved, settings: { negative_prompt: "", width: 768, height: 1152, batch_size: 1 } },
    review_digest: "sha256:4",
  };

  it("offers the picture's shape at a size the workflow makes, and makes it", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(shaped);
    vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as never);
    vi.mocked(api.remixPicture).mockResolvedValue(undefined as never);
    renderDialog(vi.fn());
    await choose();

    const offered = await screen.findByRole("checkbox", { name: "This picture's shape, at 768 × 1152" });
    expect(screen.queryByRole("checkbox", { name: "Size" })).toBeNull();
    fireEvent.click(offered);
    await waitFor(() => expect(api.remixPreview).toHaveBeenLastCalledWith(
      artifactId,
      { workflow_revision_id: "rev_cup", profile_id: "profile_cup", apply: ["shape"], role: "words" },
      expect.anything(),
    ));
    const make = screen.getByRole("button", { name: "Make this picture" });
    await waitFor(() => expect(make).toHaveAttribute("aria-disabled", "false"));
    fireEvent.click(make);

    await waitFor(() => expect(api.remixPicture).toHaveBeenCalledWith("chat_new", {
      artifact_id: artifactId,
      workflow_revision_id: "rev_cup",
      profile_id: "profile_cup",
      apply: ["shape"],
      role: "words",
      review_digest: "sha256:4",
    }));
  });
});

describe("a remix that starts from the picture itself", () => {
  const changed = {
    ...preview,
    role: "edit",
    operation: "image_to_image",
    source: { width: 512, height: 768 },
    claims: [
      claim("prompt", PROMPT, "supported"),
      claim("denoise", 0.45, "supported"),
      { ...claim("width", 512, "supported"), setting: null, applied: true },
      { ...claim("height", 768, "supported"), setting: null, applied: true },
    ],
    resolved: {
      text: PROMPT,
      settings: { negative_prompt: "", denoise: 0.6, width: 1024, height: 1024, batch_size: 1 },
      seed_drawn: false,
      trigger_words: [],
      engine_prompt: `Keep what the words do not change. Requested edit: ${PROMPT}`,
      strength: { parameter: "denoise", mode: "auto", value: 0.6, from_file: false },
    },
    review_digest: "sha256:5",
  };
  const term = (name: string) => screen.getByText(name, { selector: "dt" }).nextElementSibling?.textContent;

  it("lists workflows that change a picture, and shows what changing this one would run", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(changed);
    vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as never);
    vi.mocked(api.remixPicture).mockResolvedValue(undefined as never);
    renderDialog(vi.fn());
    fireEvent.click(await screen.findByRole("radio", { name: "This picture" }));
    await waitFor(() => expect(api.workflowReadyRevisions).toHaveBeenCalledWith(
      expect.objectContaining({ operation: "image_to_image" }), expect.anything(),
    ));
    await choose();

    await screen.findByText("A remix would change this picture with");
    expect(api.remixPreview).toHaveBeenLastCalledWith(
      artifactId, expect.objectContaining({ role: "edit", apply: [] }), expect.any(AbortSignal),
    );
    expect(term("Starts from")).toBe("this picture, 512 × 768");
    expect(term("Strength of the change")).toBe("0.6, estimated from the words");
    expect(term("Words the engine is given")).toBe(changed.resolved.engine_prompt);
    expect(screen.queryByText("Width", { selector: "dt" })).toBeNull();
    // The kept size is used as it is; only the strength can be chosen.
    expect(screen.getAllByRole("checkbox").map((box) => box.closest("label")?.textContent)).toEqual(["Denoise"]);
    const make = screen.getByRole("button", { name: "Make this picture" });
    await waitFor(() => expect(make).toHaveAttribute("aria-disabled", "false"));
    fireEvent.click(make);

    await waitFor(() => expect(api.remixPicture).toHaveBeenCalledWith(
      "chat_new", expect.objectContaining({ role: "edit", apply: [], review_digest: "sha256:5" }),
    ));
  });

  it("names a strength taken from the file as the one this picture was made with", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue({
      ...changed,
      resolved: { ...changed.resolved, strength: { parameter: "denoise", mode: "manual", value: 0.45, from_file: true } },
    });
    renderDialog();
    fireEvent.click(await screen.findByRole("radio", { name: "This picture" }));
    await choose();

    await screen.findByText("A remix would change this picture with");
    expect(term("Strength of the change")).toBe("0.45, the strength this picture was made with");
  });

  it("starts a new choice of workflow when what the remix starts from changes", async () => {
    vi.mocked(api.remixPreview).mockResolvedValue(preview);
    renderDialog();
    await choose();
    await screen.findByText("A remix would make one picture with");

    fireEvent.click(screen.getByRole("radio", { name: "This picture" }));

    expect((screen.getByLabelText("Workflow") as HTMLSelectElement).value).toBe("");
    expect((screen.getByRole("radio", { name: "This picture" }) as HTMLInputElement).checked).toBe(true);
  });
});
