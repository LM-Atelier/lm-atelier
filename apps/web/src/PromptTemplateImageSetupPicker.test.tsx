/** A template's own LoRA strengths, as their boxes are typed into. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { PromptTemplateImageSetupPicker } from "./PromptTemplateImageSetupPicker";
import { api } from "./api";
import type { SimplePromptTemplateResourcePolicy } from "./promptTemplateImageSetup";

vi.mock("./api", () => ({ api: { workflowFamilies: vi.fn(), modelAssets: vi.fn() } }));

const DIGEST = "c".repeat(64);
const POLICY: SimplePromptTemplateResourcePolicy = {
  mode: "fixed",
  workflow_revision_id: "workflow-revision-1",
  lora_policy: { mode: "fixed", stack: [{ sha256: DIGEST, model_strength: 0.8, clip_strength: 0.7 }] },
};

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("keeps a LoRA's last strength when its box is emptied, and takes the next number typed", () => {
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.modelAssets).mockResolvedValue([]);
  const onChange = vi.fn();
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <PromptTemplateImageSetupPicker value={POLICY} onChange={onChange} />
    </QueryClientProvider>,
  );
  const model = screen.getByLabelText("Template LoRA 1 model strength");
  const clip = screen.getByLabelText("Template LoRA 1 CLIP strength");

  fireEvent.change(model, { target: { value: "" } });
  fireEvent.change(clip, { target: { value: "" } });

  // Nothing that is not a strength reaches the template, and the boxes show the strengths it keeps.
  expect(onChange).not.toHaveBeenCalled();
  expect(model).toHaveValue(0.8);
  expect(clip).toHaveValue(0.7);

  fireEvent.change(model, { target: { value: "0.5" } });
  fireEvent.change(clip, { target: { value: "0.25" } });

  expect(onChange.mock.calls.map(([policy]) => policy.lora_policy.stack)).toEqual([
    [{ sha256: DIGEST, model_strength: 0.5, clip_strength: 0.7 }],
    [{ sha256: DIGEST, model_strength: 0.8, clip_strength: 0.25 }],
  ]);
});
