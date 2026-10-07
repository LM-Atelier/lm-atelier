import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import type { EngineRole, GenerationPreset } from "./types";

export function useGenerationPresetLibrary(role: EngineRole, presetIds: Array<string | null | undefined>, enabled = true, resolveDefault = true) {
  const [searches, setSearches] = useState<Partial<Record<EngineRole, string>>>({});
  const search = searches[role] ?? "";
  const selectedIds = [...new Set(presetIds.filter((id): id is string => Boolean(id)))].sort();
  const pages = useInfiniteQuery({
    queryKey: ["presets", "choices", role, search],
    queryFn: ({ pageParam }) => api.presetsPage({ role, limit: 50, offset: pageParam, search }),
    initialPageParam: 0,
    getNextPageParam: (page, previous) => page.length === 50 ? previous.length * 50 : undefined,
    enabled,
  });
  const defaults = useQuery({
    queryKey: ["presets", "default", role],
    queryFn: () => api.presetsPage({ role, limit: 1, defaultsOnly: true }),
    enabled: enabled && resolveDefault,
  });
  const selected = useQuery({
    queryKey: ["presets", "selected", role, selectedIds],
    queryFn: () => api.presetsPage({ role, limit: selectedIds.length, presetIds: selectedIds }),
    enabled: enabled && selectedIds.length > 0,
  });
  const identityRows = [
    ...(resolveDefault && defaults.isSuccess ? defaults.data : []),
    ...(selectedIds.length && selected.isSuccess ? selected.data : []),
  ];
  const read = (!resolveDefault || defaults.isSuccess) && (!selectedIds.length || selected.isSuccess);
  const missing = read && selectedIds.some((id) => !identityRows.some((preset) => preset.id === id));
  const choices = new Map<string, GenerationPreset>();
  for (const row of pages.data?.pages.flat() ?? []) choices.set(row.id, row);
  for (const row of identityRows) choices.set(row.id, row);
  return {
    search,
    setSearch: (value: string) => setSearches((current) => ({ ...current, [role]: value })),
    pages,
    choices: [...choices.values()],
    identityRows,
    ready: read && !missing,
    missing,
    pending: (resolveDefault && defaults.isPending) || (selectedIds.length > 0 && selected.isPending),
    error: (resolveDefault ? defaults.error : null) ?? (selectedIds.length ? selected.error : null),
    retry: () => { if (resolveDefault) void defaults.refetch(); if (selectedIds.length) void selected.refetch(); },
  };
}
