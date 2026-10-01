import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import type { EngineRole, ModelProfile } from "./types";

type InputModality = "text" | "image";

export function useProfileIdentity(role: EngineRole, profileId: string | null | undefined,
  enabled = true, resolveDefault = false, inputModality?: InputModality) {
  const selectedId = profileId && profileId !== "__auto__" ? profileId : null;
  const needed = enabled && Boolean(selectedId || resolveDefault);
  const read = useQuery({
    queryKey: ["profiles", "identity", role, selectedId, resolveDefault, inputModality],
    queryFn: () => api.profilesPage({ role, limit: 1, inputModality,
      ...(selectedId ? { profileIds: [selectedId] } : { defaultsOnly: true }) }),
    enabled: needed,
  });
  const profile = needed && read.isSuccess ? read.data[0] : undefined;
  const missing = needed && Boolean(selectedId) && read.isSuccess && !profile;
  return { profile, ready: !needed || (read.isSuccess && !missing), missing,
    pending: needed && read.isPending, error: needed ? read.error : null, retry: () => { void read.refetch(); } };
}

export function useProfileLibrary(role: EngineRole, profileId: string | null | undefined,
  enabled = true, resolveDefault = false, inputModality?: InputModality) {
  const [searches, setSearches] = useState<Partial<Record<EngineRole, string>>>({});
  const search = searches[role] ?? "";
  const identity = useProfileIdentity(role, profileId, enabled, resolveDefault, inputModality);
  const pages = useInfiniteQuery({
    queryKey: ["profiles", "choices", role, inputModality, search],
    queryFn: ({ pageParam }) => api.profilesPage({ role, inputModality, limit: 50, offset: pageParam, search }),
    initialPageParam: 0,
    getNextPageParam: (page, previous) => page.length === 50 ? previous.length * 50 : undefined,
    enabled,
  });
  const choices = new Map<string, ModelProfile>();
  for (const profile of pages.data?.pages.flat() ?? []) choices.set(profile.id, profile);
  if (identity.profile) choices.set(identity.profile.id, identity.profile);
  return { identity, pages, choices: [...choices.values()], search,
    setSearch: (value: string) => setSearches((current) => ({ ...current, [role]: value })) };
}
