import type { EngineRole, WorkflowSelectorCapability, WorkflowVariantReadiness } from "./types";

export interface WorkflowReadPageOptions {
  limit?: number;
  offset?: number;
  search?: string;
  operation?: string;
  role?: EngineRole;
  workflowIds?: string[];
  revisionIds?: string[];
  ungroupedOnly?: boolean;
}

export function workflowReadQuery(options: WorkflowReadPageOptions) {
  const query = new URLSearchParams();
  if (options.limit !== undefined) query.set("limit", String(options.limit));
  if (options.offset !== undefined) query.set("offset", String(options.offset));
  if (options.search) query.set("search", options.search);
  if (options.operation) query.set("operation", options.operation);
  if (options.role) query.set("role", options.role);
  if (options.ungroupedOnly) query.set("ungrouped_only", "true");
  options.workflowIds?.forEach(id => query.append("workflow_id", id));
  options.revisionIds?.forEach(id => query.append("revision_id", id));
  const value = query.toString();
  return value ? "?" + value : "";
}

export interface WorkflowFamilyReadOptions extends Pick<WorkflowReadPageOptions, "limit" | "offset" | "search" | "operation"> {
  variantLimit?: number;
  variantOffset?: number;
  variantCapability?: WorkflowSelectorCapability;
  familyIds?: string[];
  workflowIds?: string[];
  readiness?: WorkflowVariantReadiness;
  source?: "profile" | "workflow";
  order?: "name" | "readiness" | "preference";
  defaultsOnly?: boolean;
  enabledOnly?: boolean;
}

export function workflowFamilyQuery(options: WorkflowFamilyReadOptions) {
  const query = new URLSearchParams(workflowReadQuery(options).slice(1));
  if (options.variantLimit !== undefined) query.set("variant_limit", String(options.variantLimit));
  if (options.variantOffset !== undefined) query.set("variant_offset", String(options.variantOffset));
  if (options.variantCapability) query.set("variant_capability", options.variantCapability);
  if (options.readiness) query.set("readiness", options.readiness);
  if (options.source) query.set("source", options.source);
  if (options.order) query.set("order", options.order);
  if (options.defaultsOnly) query.set("defaults_only", "true");
  if (options.enabledOnly) query.set("enabled_only", "true");
  options.familyIds?.forEach(id => query.append("family_ids", id));
  query.delete("workflow_id");
  options.workflowIds?.forEach(id => query.append("workflow_ids", id));
  return query;
}
