import type { PromptTemplateResourcePolicy } from "./types";

const SHA256 = /^[0-9a-f]{64}$/;

export type SimplePromptTemplateResourcePolicy = Extract<PromptTemplateResourcePolicy, { mode: "inherited" | "fixed" }>;

export function promptTemplateImageSetupIsComplete(policy: SimplePromptTemplateResourcePolicy): boolean {
  if (policy.mode === "inherited") return true;
  if (!policy.workflow_revision_id) return false;
  return policy.lora_policy.mode !== "fixed"
    || (policy.lora_policy.stack.length > 0
      && policy.lora_policy.stack.every((lora) => SHA256.test(lora.sha256)));
}
