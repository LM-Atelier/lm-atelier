import { vi } from "vitest";
import { api } from "./api";
import type { WorkflowFamily, WorkflowSelectorCapability } from "./types";
import type { WorkflowFamilyReadOptions } from "./workflowReadQuery";

let loadFamilies: () => Promise<WorkflowFamily[]> = async () => [];
export function resetWorkflowFamilyFixtures() { loadFamilies = async () => []; }
export async function fixtureFamilyForWorkflow(id: string) {
  return (await loadFamilies()).find(family => family.variants.some(variant => variant.id === id))?.id;
}

const ranks = { ready: 0, setup_required: 1, review_required: 2, unavailable: 3 };
export function familyFixturePage(rows: WorkflowFamily[], includeArchived = false, options: WorkflowFamilyReadOptions = {}, capability?: WorkflowSelectorCapability) {
  const needle = options.search?.trim().toLocaleLowerCase() ?? "";
  const matching = rows.filter(family => (includeArchived || !family.archived)
    && (!options.enabledOnly || family.enabled)
    && ((!capability && !options.enabledOnly && !options.defaultsOnly) || family.preferences.some(preference =>
      (!capability || preference.selector_capability === capability)
      && (!options.enabledOnly || preference.enabled) && (!options.defaultsOnly || preference.is_default)))
    && (!options.familyIds || options.familyIds.includes(family.id))
    && (!options.workflowIds || family.variants.some(variant => options.workflowIds!.includes(variant.id)))
    && (!options.source || family.compatibility === (options.source === "profile"))
    && (!options.defaultsOnly || family.preferences.some(preference => preference.is_default))
    && (!needle || [family.name, family.description, family.use_case, ...family.tags,
      ...(family.dependency_summary?.names ?? []), ...family.variants.map(variant => variant.name)]
      .some(value => value.toLocaleLowerCase().includes(needle))))
    .map(family => {
      const variants = family.variants.filter(variant => (!options.operation || variant.operation === options.operation)
        && (!options.readiness || variant.readiness === options.readiness)
        && (!options.workflowIds || options.workflowIds.includes(variant.id)));
      const readiness = variants.reduce<keyof typeof ranks>((best, variant) =>
        ranks[variant.readiness] < ranks[best] ? variant.readiness : best, "unavailable");
      const selectorOperations: Record<WorkflowSelectorCapability, string[]> = {
        chat: ["text"], image: ["text_to_image", "image_to_image"],
        video: ["text_to_video", "image_to_video"], vision: [],
      };
      const supported = (Object.keys(selectorOperations) as WorkflowSelectorCapability[])
        .filter(capability => family.variants.some(variant => selectorOperations[capability].includes(variant.operation)));
      return { ...family, supported_selector_capabilities: family.supported_selector_capabilities ?? supported,
        variant_count: variants.length,
        ready_variant_count: variants.filter(variant => variant.readiness === "ready").length,
        best_readiness: readiness,
        variants: variants.slice(options.variantOffset ?? 0,
          options.variantLimit === undefined ? undefined : (options.variantOffset ?? 0) + options.variantLimit) };
    }).filter(family => options.familyIds || options.workflowIds || family.variant_count > 0)
    .sort((a, b) => (options.order === "readiness" ? ranks[a.best_readiness] - ranks[b.best_readiness] : 0)
      || a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  return matching.slice(options.offset ?? 0, options.limit === undefined ? undefined : (options.offset ?? 0) + options.limit);
}

export function mockWorkflowFamilyPages(rows: WorkflowFamily[] | (() => Promise<WorkflowFamily[]>)) {
  loadFamilies = typeof rows === "function" ? rows : async () => rows;
  vi.mocked(api.workflowFamilies).mockImplementation(async (capability, archived, _dependencies, options) =>
    familyFixturePage(await loadFamilies(), archived, options, capability));
  vi.mocked(api.workflowFamilyOperations).mockImplementation(async archived => [...new Set(
    (await loadFamilies()).filter(family => archived || !family.archived)
      .flatMap(family => family.variants.map(variant => variant.operation)),
  )].sort());
}
